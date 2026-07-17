#!/usr/bin/env python3
"""Convert a LeRobot v2/v2.1 dataset to Dexdata format.

The converter keeps videos in place and writes one JSONL file per episode.  It
uses the explicit LeRobot action column when present, so OpenDM can build action
chunks without approximating controls from future observations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress display
    def tqdm(iterable, **_kwargs):
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="LeRobot dataset root")
    parser.add_argument("output_dir", type=Path, help="Directory for OpenDM JSONL files")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument(
        "--video-keys",
        nargs="+",
        help="LeRobot video feature keys (default: discover all video features)",
    )
    parser.add_argument(
        "--prompt",
        help="Override prompts for every frame; otherwise use LeRobot task metadata",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--episodes", nargs="+", type=int, help="Only convert these episode IDs")
    parser.add_argument("--max-episodes", type=int, help="Convert at most this many episodes")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-action",
        action="store_true",
        help="Do not emit action; OpenDM will use the next state as the target",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def as_list(value: Any, key: str) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be a vector, got {type(value).__name__}")
    return [float(item) for item in value]


def discover_video_keys(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]


def format_dataset_path(template: str, episode_index: int, chunks_size: int) -> str:
    return template.format(
        episode_index=episode_index,
        episode_chunk=episode_index // chunks_size,
    )


def task_maps(root: Path) -> tuple[dict[int, str], dict[int, str]]:
    tasks = {
        int(row["task_index"]): str(row["task"])
        for row in read_jsonl(root / "meta" / "tasks.jsonl")
    }
    episodes = {}
    for row in read_jsonl(root / "meta" / "episodes.jsonl"):
        values = row.get("tasks") or []
        if values:
            episodes[int(row["episode_index"])] = str(values[0])
    return tasks, episodes


def iter_episode_ids(root: Path, info: dict[str, Any]) -> Iterable[int]:
    episode_rows = read_jsonl(root / "meta" / "episodes.jsonl")
    if episode_rows:
        return sorted(int(row["episode_index"]) for row in episode_rows)
    return range(int(info["total_episodes"]))


def write_jsonl_atomic(path: Path, samples: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for sample in samples:
                stream.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit(
            "pyarrow is required to read LeRobot parquet files; install the OpenDM "
            "dependencies or run `pip install pyarrow`."
        ) from error
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be positive")

    root = args.input_dir.resolve()
    output = args.output_dir.resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
    info = read_json(info_path)
    version = str(info.get("codebase_version", ""))
    if version and not (version.startswith("v2") or version.startswith("2")):
        raise ValueError(f"expected LeRobot v2/v2.1, found {version!r}")

    video_keys = args.video_keys or discover_video_keys(info)
    if not video_keys:
        raise ValueError("no video features found; pass --video-keys explicitly")
    unknown = sorted(set(video_keys) - set(info.get("features", {})))
    if unknown:
        raise ValueError(f"unknown video feature(s): {', '.join(unknown)}")

    episode_ids = list(iter_episode_ids(root, info))
    if args.episodes is not None:
        requested = set(args.episodes)
        missing = sorted(requested - set(episode_ids))
        if missing:
            raise ValueError(f"unknown episode IDs: {missing}")
        episode_ids = [item for item in episode_ids if item in requested]
    if args.max_episodes is not None:
        episode_ids = episode_ids[: args.max_episodes]

    chunks_size = int(info.get("chunks_size", 1000))
    data_template = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    video_template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    task_by_index, task_by_episode = task_maps(root)
    output.mkdir(parents=True, exist_ok=True)
    index: dict[str, int] = {}

    for episode_id in tqdm(episode_ids, desc="Converting episodes", unit="episode"):
        destination = output / f"episode_{episode_id:06d}.jsonl"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"output exists (use --overwrite): {destination}")
        parquet_path = root / format_dataset_path(data_template, episode_id, chunks_size)
        if not parquet_path.is_file():
            raise FileNotFoundError(parquet_path)

        columns = [args.state_key, "frame_index"]
        if not args.no_action:
            columns.append(args.action_key)
        schema_names = set(pq.read_schema(parquet_path).names)
        if "task_index" in schema_names and args.prompt is None:
            columns.append("task_index")
        missing_columns = sorted(set(columns) - schema_names)
        if missing_columns:
            raise ValueError(f"missing columns in {parquet_path}: {missing_columns}")
        rows = pq.read_table(parquet_path, columns=columns).to_pylist()

        video_paths = {}
        for image_number, video_key in enumerate(video_keys, start=1):
            relative = format_dataset_path(
                video_template.replace("{video_key}", video_key), episode_id, chunks_size
            )
            video_path = root / relative
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            video_paths[f"images_{image_number}"] = relative

        samples = []
        for row_number, row in enumerate(rows):
            frame_index = int(row.get("frame_index", row_number))
            if frame_index % args.frame_stride:
                continue
            prompt = args.prompt or task_by_index.get(int(row.get("task_index", -1)))
            prompt = prompt or task_by_episode.get(episode_id)
            if prompt is None:
                raise ValueError(f"no prompt for episode {episode_id}; pass --prompt")
            sample = {
                key: {"type": "video", "url": value, "frame_idx": frame_index}
                for key, value in video_paths.items()
            }
            sample.update(
                state=as_list(row[args.state_key], args.state_key),
                prompt=prompt,
                is_robot=True,
            )
            if not args.no_action:
                sample["action"] = as_list(row[args.action_key], args.action_key)
            samples.append(sample)
        if len(samples) < 2:
            raise ValueError(f"episode {episode_id} has fewer than two selected frames")
        write_jsonl_atomic(destination, samples)
        index[str(destination)] = len(samples)

    index_path = output / "index_cache.json"
    temporary = index_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump({"data": index}, stream, indent=2)
    os.replace(temporary, index_path)
    manifest = {
        "source": str(root),
        "format": "opendm-jsonl",
        "episodes": len(index),
        "samples": sum(index.values()),
        "image_keys": [f"images_{i}" for i in range(1, len(video_keys) + 1)],
        "lerobot_video_keys": video_keys,
        "state_key": args.state_key,
        "action_key": None if args.no_action else args.action_key,
    }
    with (output / "conversion_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
