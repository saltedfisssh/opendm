#!/usr/bin/env python3
"""Convert RoboChallenge Table30 v2 into OpenDM's JSONL format.

The converter does not copy or decode videos.  Each generated sample points to a
frame in the original MP4, so conversion is inexpensive in disk space.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Iterator, TextIO

from tqdm import tqdm

EMBODIMENTS = ("arx5", "ur5", "aloha", "dos_w1")
CAMERAS = {
    "arx5": ("cam_global_rgb.mp4", "cam_arm_rgb.mp4", "cam_side_rgb.mp4"),
    "ur5": ("cam_global_rgb.mp4", "cam_arm_rgb.mp4"),
    "aloha": (
        "cam_high_rgb.mp4",
        "cam_left_wrist_rgb.mp4",
        "cam_right_wrist_rgb.mp4",
    ),
    "dos_w1": (
        "cam_high_rgb.mp4",
        "cam_left_wrist_rgb.mp4",
        "cam_right_wrist_rgb.mp4",
    ),
}
STATE_FILES = {
    "arx5": ("states.jsonl",),
    "ur5": ("states.jsonl",),
    "aloha": ("left_states.jsonl", "right_states.jsonl"),
    "dos_w1": ("left_states.jsonl", "right_states.jsonl"),
}
IMAGE_KEYS = {
    embodiment: [f"images_{index + 1}" for index in range(len(cameras))]
    for embodiment, cameras in CAMERAS.items()
}
STATE_DIMS = {"arx5": 7, "ur5": 7, "aloha": 14, "dos_w1": 14}


def parse_gripper_thresholds(values: list[str] | None) -> dict[str, tuple[float, ...]]:
    """Parse repeatable ``EMBODIMENT=t[,t]`` gripper threshold arguments."""
    result: dict[str, tuple[float, ...]] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                f"invalid gripper threshold {value!r}; expected EMBODIMENT=t[,t]"
            )
        embodiment, raw_thresholds = value.split("=", 1)
        embodiment = embodiment.strip().lower()
        if embodiment not in EMBODIMENTS:
            raise ValueError(f"unknown embodiment {embodiment!r} in gripper threshold")
        if embodiment in result:
            raise ValueError(f"duplicate gripper threshold for {embodiment}")
        try:
            thresholds = tuple(
                float(item.strip()) for item in raw_thresholds.split(",")
            )
        except ValueError as error:
            raise ValueError(
                f"invalid numeric gripper threshold for {embodiment}: "
                f"{raw_thresholds!r}"
            ) from error
        expected = len(STATE_FILES[embodiment])
        if len(thresholds) != expected:
            raise ValueError(
                f"{embodiment} requires {expected} gripper threshold(s), "
                f"got {len(thresholds)}"
            )
        if not all(math.isfinite(threshold) for threshold in thresholds):
            raise ValueError(f"gripper thresholds for {embodiment} must be finite")
        result[embodiment] = thresholds
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/Table30v2"),
        help="Table30v2 root (default: data/Table30v2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/table30v2_dexdata"),
        help="Output annotation root (default: data/table30v2_dexdata)",
    )
    parser.add_argument(
        "--embodiments",
        nargs="+",
        choices=EMBODIMENTS,
        default=list(EMBODIMENTS),
        help="Embodiments to convert",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional task-name allowlist (default: all tasks)",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Keep every Nth source frame (default: 1)",
    )
    parser.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=None,
        help="Limit episodes per task, useful for a smoke test",
    )
    parser.add_argument(
        "--gripper-threshold",
        action="append",
        default=[],
        metavar="EMBODIMENT=t[,t]",
        help=(
            "Binarize gripper widths as 0=closed and 1=open using an "
            "embodiment-specific threshold. Repeat for multiple embodiments; "
            "dual-arm robots require left,right thresholds. Example: "
            "--gripper-threshold arx5=0.03 --gripper-threshold "
            "aloha=0.0406,0.0345"
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing episode JSONL files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and count episodes without writing",
    )
    args = parser.parse_args()
    if args.frame_interval < 1:
        parser.error("--frame-interval must be >= 1")
    if args.max_episodes_per_task is not None and args.max_episodes_per_task < 1:
        parser.error("--max-episodes-per-task must be >= 1")
    try:
        args.gripper_thresholds = parse_gripper_thresholds(args.gripper_threshold)
    except ValueError as error:
        parser.error(str(error))
    unused_thresholds = set(args.gripper_thresholds) - set(args.embodiments)
    if unused_thresholds:
        parser.error(
            "gripper threshold configured for unselected embodiment(s): "
            + ", ".join(sorted(unused_thresholds))
        )
    return args


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def embodiment_from_episode(task_info: dict, episode_meta: dict) -> str:
    robot_id = str(episode_meta.get("robot_id", "")).lower()
    tags = " ".join(
        map(str, task_info.get("task_desc", {}).get("task_tag", []))
    ).lower()
    identity = f"{robot_id} {tags}"
    if "arx5" in identity:
        return "arx5"
    if "ur5" in identity:
        return "ur5"
    if "aloha" in identity:
        return "aloha"
    if "w1" in identity or "dos-w1" in identity or "dos_w1" in identity:
        return "dos_w1"
    raise ValueError(
        f"cannot identify embodiment from robot_id={robot_id!r}, tags={tags!r}"
    )


def state_lines(paths: list[Path]) -> Iterator[tuple[int, list[dict]]]:
    """Yield synchronized state rows, rejecting truncated bimanual streams."""
    with ExitStack() as stack:
        streams: list[TextIO] = [
            stack.enter_context(path.open("r", encoding="utf-8")) for path in paths
        ]
        frame_idx = 0
        while True:
            lines = [stream.readline() for stream in streams]
            if not any(lines):
                return
            if not all(lines):
                raise ValueError(f"state streams have different lengths: {paths}")
            yield frame_idx, [json.loads(line) for line in lines]
            frame_idx += 1


def make_state(
    rows: list[dict],
    paths: list[Path],
    gripper_thresholds: tuple[float, ...] | None = None,
) -> list[float]:
    if gripper_thresholds is not None and len(gripper_thresholds) != len(rows):
        raise ValueError(
            f"expected {len(rows)} gripper threshold(s), "
            f"got {len(gripper_thresholds)}"
        )
    state: list[float] = []
    for arm_index, (row, path) in enumerate(zip(rows, paths, strict=True)):
        joints = row.get("joint_positions")
        if not isinstance(joints, list) or len(joints) != 6:
            raise ValueError(f"expected 6 joint_positions in {path}")
        gripper = row.get("gripper_width")
        if isinstance(gripper, list):
            if len(gripper) != 1:
                raise ValueError(f"expected scalar gripper_width in {path}")
            gripper = gripper[0]
        if not isinstance(gripper, (int, float)):
            raise ValueError(f"invalid gripper_width in {path}")
        state.extend(float(value) for value in joints)
        if gripper_thresholds is None:
            state.append(float(gripper))
        else:
            # Width grows as the gripper opens on all Table30v2 embodiments.
            state.append(float(gripper >= gripper_thresholds[arm_index]))
    return state


def episode_output_name(task_name: str, episode_name: str) -> str:
    del task_name
    return f"{episode_name}.jsonl"


def write_index(path: Path, files: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump({"data": files}, output, indent=2)
    os.replace(temporary, path)


def load_index(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    data = read_json(path).get("data")
    if not isinstance(data, dict):
        raise ValueError(f"invalid index cache: {path}")
    return {str(key): int(value) for key, value in data.items()}


def rebuild_outputs(
    output_dir: Path,
    raw_dir: Path,
    converted_indexes: dict[str, dict[str, dict[str, int]]],
    gripper_thresholds: dict[str, tuple[float, ...]] | None = None,
) -> dict:
    """Update per-task/embodiment indexes and write the registration manifest."""
    datasets = {}
    gripper_thresholds = gripper_thresholds or {}
    for embodiment in EMBODIMENTS:
        embodiment_dir = output_dir / embodiment
        if not embodiment_dir.is_dir():
            continue

        aggregate: dict[str, int] = {}
        task_names = []
        for task_dir in sorted(
            path for path in embodiment_dir.iterdir() if path.is_dir()
        ):
            index_path = task_dir / "index_cache.json"
            updates = converted_indexes.get(embodiment, {}).get(task_dir.name)
            if updates is not None:
                current = load_index(index_path)
                current.update(updates)
                # Drop entries whose annotation was removed outside this script.
                current = {
                    path: count
                    for path, count in current.items()
                    if Path(path).is_file()
                }
                write_index(index_path, current)
            task_files = load_index(index_path)
            if not task_files:
                continue
            aggregate.update(task_files)
            task_names.append(task_dir.name)
            dataset_name = f"table30v2_{embodiment}_{task_dir.name}"
            datasets[dataset_name] = {
                "kind": "single_task",
                "embodiment": embodiment,
                "tasks": [task_dir.name],
                "jsonl_dir": str(task_dir),
                "image_dir": str(raw_dir),
                "image_keys": IMAGE_KEYS[embodiment],
                "state_dim": STATE_DIMS[embodiment],
            }
            if embodiment in gripper_thresholds:
                datasets[dataset_name]["gripper_binarization"] = {
                    "thresholds": list(gripper_thresholds[embodiment]),
                    "closed_value": 0.0,
                    "open_value": 1.0,
                }

        if not aggregate:
            continue
        write_index(embodiment_dir / "index_cache.json", aggregate)
        dataset_name = f"table30v2_{embodiment}"
        datasets[dataset_name] = {
            "kind": "multi_task",
            "embodiment": embodiment,
            "tasks": task_names,
            "jsonl_dir": str(embodiment_dir),
            "image_dir": str(raw_dir),
            "image_keys": IMAGE_KEYS[embodiment],
            "state_dim": STATE_DIMS[embodiment],
        }
        if embodiment in gripper_thresholds:
            datasets[dataset_name]["gripper_binarization"] = {
                "thresholds": list(gripper_thresholds[embodiment]),
                "closed_value": 0.0,
                "open_value": 1.0,
            }

    manifest = {
        "format_version": 1,
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "datasets": datasets,
        "gripper_binarization": {
            embodiment: {
                "thresholds": list(thresholds),
                "closed_value": 0.0,
                "open_value": 1.0,
            }
            for embodiment, thresholds in gripper_thresholds.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
    os.replace(temporary, manifest_path)
    return manifest


def convert_episode(
    episode_dir: Path,
    task_name: str,
    prompt: str,
    embodiment: str,
    raw_dir: Path,
    output_file: Path,
    frame_interval: int,
    overwrite: bool,
    gripper_thresholds: tuple[float, ...] | None = None,
) -> int:
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"output exists (use --overwrite): {output_file}")

    videos = [episode_dir / "videos" / name for name in CAMERAS[embodiment]]
    states = [episode_dir / "states" / name for name in STATE_FILES[embodiment]]
    missing = [path for path in videos + states if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing episode files: {', '.join(map(str, missing))}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    samples = 0
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for frame_idx, rows in state_lines(states):
                if frame_idx % frame_interval:
                    continue
                sample = {
                    f"images_{index + 1}": {
                        "type": "video",
                        "url": os.path.relpath(video, raw_dir),
                        "frame_idx": frame_idx,
                    }
                    for index, video in enumerate(videos)
                }
                sample.update(
                    state=make_state(rows, states, gripper_thresholds),
                    prompt=prompt,
                    is_robot=True,
                )
                output.write(
                    json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                samples += 1
        if samples < 2:
            raise ValueError(f"episode has fewer than 2 selected frames: {episode_dir}")
        os.replace(temporary, output_file)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return samples


def main() -> int:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw dataset directory not found: {raw_dir}")

    task_dirs = sorted(path for path in raw_dir.iterdir() if (path / "data").is_dir())
    available_tasks = {path.name for path in task_dirs}
    if args.tasks:
        unknown = sorted(set(args.tasks) - available_tasks)
        if unknown:
            raise ValueError(f"unknown tasks: {', '.join(unknown)}")
        task_dirs = [path for path in task_dirs if path.name in set(args.tasks)]

    counts = {name: {"episodes": 0, "samples": 0} for name in args.embodiments}
    index_data: dict[str, dict[str, dict[str, int]]] = {
        name: {} for name in args.embodiments
    }
    for task_dir in tqdm(task_dirs, desc="Tasks", unit="task"):
        task_info = read_json(task_dir / "meta" / "task_info.json")
        prompt = task_info.get("task_desc", {}).get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"missing prompt in {task_dir / 'meta/task_info.json'}")
        episodes = sorted(
            path for path in (task_dir / "data").iterdir() if path.is_dir()
        )
        if args.max_episodes_per_task is not None:
            episodes = episodes[: args.max_episodes_per_task]
        episode_iter = tqdm(
            episodes,
            desc=f"{task_dir.name} episodes",
            unit="episode",
            leave=False,
        )
        for episode_dir in episode_iter:
            meta_path = episode_dir / "meta" / "episode_meta.json"
            embodiment = embodiment_from_episode(task_info, read_json(meta_path))
            if embodiment not in counts:
                continue
            counts[embodiment]["episodes"] += 1
            if args.dry_run:
                continue
            output_file = (
                output_dir
                / embodiment
                / task_dir.name
                / episode_output_name(task_dir.name, episode_dir.name)
            )
            samples = convert_episode(
                episode_dir,
                task_dir.name,
                prompt,
                embodiment,
                raw_dir,
                output_file,
                args.frame_interval,
                args.overwrite,
                args.gripper_thresholds.get(embodiment),
            )
            counts[embodiment]["samples"] += samples
            task_index = index_data[embodiment].setdefault(task_dir.name, {})
            task_index[str(output_file)] = samples

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = rebuild_outputs(
            output_dir, raw_dir, index_data, args.gripper_thresholds
        )

    for embodiment in args.embodiments:
        values = counts[embodiment]
        print(
            f"{embodiment}: {values['episodes']} episodes, {values['samples']} samples"
        )
        if embodiment in args.gripper_thresholds:
            thresholds = ", ".join(
                str(value) for value in args.gripper_thresholds[embodiment]
            )
            print(
                f"  gripper binarization: threshold(s) [{thresholds}], "
                "closed=0, open=1"
            )
    if args.dry_run:
        print("Dry run only; no files were written.")
    else:
        print(f"Converted annotations: {output_dir}")
        print(f"Use this as image_dir: {raw_dir}")
        print("Registered dataset names:")
        for name, config in sorted(manifest["datasets"].items()):
            print(f"  {name} ({config['kind']}, {len(config['tasks'])} task(s))")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
