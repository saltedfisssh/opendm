#!/usr/bin/env python3
"""Convert the Piper bimanual LeRobot capture into OpenDM training JSONL.

One ``.jsonl`` per episode, one JSON object per frame, as specified in
``docs/en/data.md``. Videos are **not** copied: frames are referenced in place
with ``{"type": "video", "url": ..., "frame_idx": ...}`` against the read-only
capture mount, which random-seek decodes in about 2 ms per frame.

Representations
---------------
``joint``
    ``[L j1..j6, L gripper, R j1..j6, R gripper]`` -- the recorded joint angles,
    verbatim. 14 dims.
``eef_local``
    Per arm ``[xyz, axis-angle, gripper]`` in that arm's **own** base frame.
    14 dims.
``eef_unified``
    Same layout, but the right arm is mapped into the left arm's base frame, so
    both arms live in one frame. 14 dims.
``eef_unified_pair``
    ``eef_unified`` plus the inter-gripper relative pose ``T_left^-1 @ T_right``
    as ``[xyz, rot6d]``. 23 dims. The extra block is redundant in principle, but
    DM05 consumes state as discretised text tokens, so composing it internally
    from binned values is not something the model can be expected to do.

Output layout, per representation::

    <out-root>/<representation>/jsonl/train/episode_%06d.jsonl
    <out-root>/<representation>/jsonl/held_out/episode_%06d.jsonl

Train and held-out episodes go into separate directories because
``JsonlDataset`` globs recursively and has no notion of a split. The split is
drawn once into ``<out-root>/split.json`` and reused by every representation, so
all rungs of the ablation see identical episodes.

End-effector poses are recomputed from the joint angles with
:func:`opendm.kinematics.piper.fk` rather than read from ``observation.qpos_ee``.
Both agree to a few micrometres on the median frame, but the capture samples
joint and end-pose CAN feedback separately, so during fast motion they disagree
by up to ~6 mm. Recomputing keeps every representation describing one identical
trajectory, which is required for a representation ablation to be attributable.

The action field is omitted on purpose: ``action[t] == observation.state[t+1]``
holds exactly in this capture, so OpenDM's "use future state as target" path in
``BuildActionChunk`` produces identical targets with less on-disk duplication.

Usage::

    python script/piper_lerobot_to_jsonl.py \\
        --source /mnt/xiaoyu_teleop_data/piper/20260907/piper_fold_cloth_in_place \\
        --out-root ./data/piper_fold_cloth \\
        --representation joint eef_local eef_unified eef_unified_pair
"""

import argparse
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow.parquet as pq

from opendm.data import se3
from opendm.kinematics import piper

REPRESENTATIONS = ("joint", "eef_local", "eef_unified", "eef_unified_pair")

# Camera keys in the capture, mapped to the image field order OpenDM expects.
CAMERA_KEYS = (
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
)
IMAGE_FIELDS = ("images_1", "images_2", "images_3")

# ``observation.state`` layout: 6 joints + gripper per arm.
LEFT_JOINTS = slice(0, 6)
LEFT_GRIPPER = 6
RIGHT_JOINTS = slice(7, 13)
RIGHT_GRIPPER = 13

HELD_OUT_FRACTION = 0.05


def _episode_poses(joints: np.ndarray) -> np.ndarray:
    """Flange poses for one arm over an episode, with a continuous rotation chart.

    Args:
        joints: ``(T, 6)`` joint angles in radians.

    Returns:
        ``(T, 4, 4)`` flange poses in that arm's base frame.
    """
    return piper.fk(joints)


def _pos_rotvec(poses: np.ndarray) -> np.ndarray:
    """``(T, 4, 4)`` poses to ``(T, 6)`` ``[xyz, axis-angle]``, unwrapped in time."""
    positions = poses[:, :3, 3]
    rotvec = se3.unwrap_rotvec_sequence(se3.mat_to_rotvec(poses[:, :3, :3]))
    return np.concatenate([positions, rotvec], axis=-1)


def build_states(state: np.ndarray, representation: str) -> np.ndarray:
    """Build the per-frame state vectors for one episode.

    Args:
        state: ``(T, 14)`` recorded ``observation.state``.
        representation: One of :data:`REPRESENTATIONS`.

    Returns:
        ``(T, D)`` state array, ``float64``.
    """
    if representation == "joint":
        return state.astype(np.float64)

    left = _episode_poses(state[:, LEFT_JOINTS])
    right = _episode_poses(state[:, RIGHT_JOINTS])
    left_gripper = state[:, LEFT_GRIPPER : LEFT_GRIPPER + 1]
    right_gripper = state[:, RIGHT_GRIPPER : RIGHT_GRIPPER + 1]

    if representation in ("eef_unified", "eef_unified_pair"):
        right = piper.T_RIGHT_BASE_TO_LEFT_BASE @ right

    blocks = [
        _pos_rotvec(left),
        left_gripper,
        _pos_rotvec(right),
        right_gripper,
    ]

    if representation == "eef_unified_pair":
        # Expressed in the left gripper's own frame, so it is independent of the
        # world frame and of the (here known, generally unknown) base offset.
        blocks.append(se3.transform_to_pos_rot6d(se3.relative_transform(left, right)))

    return np.concatenate(blocks, axis=-1)


def state_dim(representation: str) -> int:
    """Return the state width a representation produces."""
    return 23 if representation == "eef_unified_pair" else 14


def _video_field(episode_index: int, camera_key: str, frame_index: int) -> dict:
    chunk = episode_index // 1000
    return {
        "type": "video",
        "url": f"videos/chunk-{chunk:03d}/{camera_key}/episode_{episode_index:06d}.mp4",
        "frame_idx": int(frame_index),
    }


def convert_episode(
    parquet_path: pathlib.Path,
    episode_index: int,
    prompt: str,
    representation: str,
    out_path: pathlib.Path,
) -> tuple[int, int]:
    """Write one episode's JSONL file.

    Returns:
        ``(num_frames, num_rotation_unwraps)`` for reporting.
    """
    table = pq.read_table(parquet_path, columns=["observation.state"]).to_pydict()
    state = np.asarray(table["observation.state"], dtype=np.float64)

    states = build_states(state, representation)

    # Count how often the continuous-chart fix actually fired, so the caller can
    # report whether the pi crossing is real on this capture or only theoretical.
    unwraps = 0
    if representation != "joint":
        raw = np.concatenate(
            [
                se3.mat_to_rotvec(_episode_poses(state[:, LEFT_JOINTS])[:, :3, :3]),
                se3.mat_to_rotvec(_episode_poses(state[:, RIGHT_JOINTS])[:, :3, :3]),
            ],
            axis=-1,
        )
        unwraps = int((np.abs(np.diff(raw, axis=0)) > np.pi).any(axis=-1).sum())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        for frame_index, vector in enumerate(states):
            frame = {
                field: _video_field(episode_index, camera, frame_index)
                for field, camera in zip(IMAGE_FIELDS, CAMERA_KEYS, strict=True)
            }
            frame["state"] = [round(float(v), 6) for v in vector]
            frame["prompt"] = prompt
            handle.write(json.dumps(frame, separators=(",", ":")) + "\n")

    return len(states), unwraps


def _worker(args: tuple) -> tuple[int, int, int]:
    episode_index, parquet_path, prompt, representation, out_path = args
    frames, unwraps = convert_episode(
        pathlib.Path(parquet_path),
        episode_index,
        prompt,
        representation,
        pathlib.Path(out_path),
    )
    return episode_index, frames, unwraps


def load_manifest(
    source: pathlib.Path,
) -> tuple[list[tuple[int, pathlib.Path, str]], int]:
    """Read the LeRobot metadata and resolve every episode's parquet and prompt."""
    info = json.loads((source / "meta" / "info.json").read_text())

    tasks = {}
    with open(source / "meta" / "tasks.jsonl") as handle:
        for line in handle:
            record = json.loads(line)
            tasks[record["task_index"]] = record["task"]

    episodes = []
    with open(source / "meta" / "episodes.jsonl") as handle:
        for line in handle:
            record = json.loads(line)
            index = record["episode_index"]
            chunk = index // int(info.get("chunks_size", 1000))
            path = source / f"data/chunk-{chunk:03d}/episode_{index:06d}.parquet"
            prompt = record["tasks"][0] if record.get("tasks") else tasks[0]
            episodes.append((index, path, prompt))

    return episodes, int(info["fps"])


def write_split(out_root: pathlib.Path, episode_indices: list[int], seed: int) -> dict:
    """Create or reuse the train/held-out episode split shared by all rungs.

    The split is written once and reused, so every rung of the ablation is
    trained and scored on exactly the same episodes.
    """
    split_path = out_root / "split.json"
    if split_path.exists():
        return json.loads(split_path.read_text())

    order = np.random.default_rng(seed).permutation(np.asarray(episode_indices))
    num_held_out = max(1, int(round(len(order) * HELD_OUT_FRACTION)))
    split = {
        "seed": seed,
        "held_out_fraction": HELD_OUT_FRACTION,
        "held_out": sorted(int(i) for i in order[:num_held_out]),
        "train": sorted(int(i) for i in order[num_held_out:]),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2) + "\n")
    return split


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=pathlib.Path(
            "/mnt/xiaoyu_teleop_data/piper/20260907/piper_fold_cloth_in_place"
        ),
        help="LeRobot v2.1 dataset root (read only).",
    )
    parser.add_argument(
        "--out-root",
        type=pathlib.Path,
        default=pathlib.Path("./data/piper_fold_cloth"),
        help="Destination root; one subdirectory per representation.",
    )
    parser.add_argument(
        "--representation",
        nargs="+",
        choices=REPRESENTATIONS,
        default=list(REPRESENTATIONS),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N episodes (for smoke tests).",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not (args.source / "meta" / "info.json").exists():
        print(f"error: no LeRobot metadata under {args.source}", file=sys.stderr)
        return 1

    episodes, fps = load_manifest(args.source)
    if args.limit is not None:
        episodes = episodes[: args.limit]

    split = write_split(
        args.out_root, [index for index, _, _ in episodes], args.split_seed
    )
    held_out = set(split["held_out"])
    print(
        f"{len(episodes)} episodes @ {fps} fps | "
        f"{len(split['train'])} train / {len(split['held_out'])} held out"
    )

    for representation in args.representation:
        jsonl_root = args.out_root / representation / "jsonl"
        jobs = [
            (
                index,
                str(path),
                prompt,
                representation,
                str(
                    jsonl_root
                    / ("held_out" if index in held_out else "train")
                    / f"episode_{index:06d}.jsonl"
                ),
            )
            for index, path, prompt in episodes
        ]

        total_frames = 0
        total_unwraps = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_worker, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), start=1):
                _, frames, unwraps = future.result()
                total_frames += frames
                total_unwraps += unwraps
                if done % 200 == 0 or done == len(futures):
                    print(
                        f"  {representation}: {done}/{len(futures)} episodes",
                        flush=True,
                    )

        # A stale index cache would silently pin the old frame count.
        for split_name in ("train", "held_out"):
            cache = jsonl_root / split_name / "index_cache.json"
            if cache.exists():
                cache.unlink()

        print(
            f"  {representation}: {total_frames} frames, "
            f"state dim {state_dim(representation)}, "
            f"rotation unwraps applied on {total_unwraps} frames"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
