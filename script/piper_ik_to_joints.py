#!/usr/bin/env python3
"""Build the S5 dataset: joints recovered by IK from an estimated virtual base.

This is the second half of rung S5. :mod:`script.piper_estimate_root` chooses a
virtual base per arm from end-effector trajectories alone; this script applies
it, converting every episode's poses back into joint angles with sequential
warm-started IK, and writes them in the same layout as the S0 joint dataset.

Because the estimated base is not the true base (it cannot be -- see the
estimator's own documentation), the resulting joint trajectories are *not* the
recorded ones. That is the point: they are what a UMI-style pipeline would have
to train on. What matters is that they are self-consistent, smooth, and stay
inside the joint limits, so a policy trained on them can be deployed through the
same virtual base.

The report at the end is the gate for whether S5 is worth training:

* ``branch flips`` -- IK jumping between elbow/wrist configurations injects
  discontinuities into the targets. Above 5% the joint stream is not trackable
  and S5 would be testing a broken conversion rather than a representation.
* ``unreached`` -- poses the estimated base cannot actually reach.
* ``near-singular`` -- frames where the Jacobian is ill-conditioned, so a small
  pose error becomes a large joint error at deployment.

Usage::

    python script/piper_estimate_root.py            # writes estimated_bases.json
    python script/piper_ik_to_joints.py --workers 32
"""

import argparse
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from opendm.data import se3
from opendm.kinematics import piper

ARM_SLICES = {"left": slice(0, 6), "right": slice(7, 13)}
GRIPPER_INDEX = {"left": 6, "right": 13}

SOURCE_REPRESENTATION = "eef_unified"
TARGET_REPRESENTATION = "joint_estimated_base"


def base_transform(estimate: dict) -> np.ndarray:
    """Build the ``4x4`` virtual base transform from an estimator record."""
    return se3.make_transform(
        np.asarray(estimate["position"], dtype=np.float64),
        se3.rotvec_to_mat(np.array([0.0, 0.0, estimate["yaw_rad"]])),
    )


def convert_episode(
    source_path: pathlib.Path,
    target_path: pathlib.Path,
    bases: dict,
) -> dict:
    """Convert one episode's unified-frame poses into joints via IK.

    Returns:
        Per-episode diagnostics for the aggregate report.
    """
    records = [json.loads(line) for line in open(source_path)]
    states = np.asarray([record["state"] for record in records], dtype=np.float64)

    joints = {}
    diagnostics = {}
    for arm, arm_slice in ARM_SLICES.items():
        world = se3.pos_rotvec_to_transform(states[:, arm_slice])
        local = se3.transform_inverse(bases[arm]) @ world

        solved, pos_err, rot_err, jumps = piper.ik_track(
            local, piper.neutral_joints(), max_iters=80, restarts=4
        )
        joints[arm] = solved
        diagnostics[arm] = {
            "frames": int(len(solved)),
            "pos_err_sum": float(pos_err.sum()),
            "rot_err_sum": float(rot_err.sum()),
            "unreached": int((pos_err > 2e-3).sum()),
            "branch_jumps": int(jumps.sum()),
            "near_singular": int((piper.min_singular_value(solved) < 1e-2).sum()),
        }

    # Same 14-dim layout as the S0 joint dataset: 6 joints + gripper per arm.
    vectors = np.concatenate(
        [
            joints["left"],
            states[:, GRIPPER_INDEX["left"] : GRIPPER_INDEX["left"] + 1],
            joints["right"],
            states[:, GRIPPER_INDEX["right"] : GRIPPER_INDEX["right"] + 1],
        ],
        axis=-1,
    )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w") as handle:
        for record, vector in zip(records, vectors, strict=True):
            record["state"] = [round(float(v), 6) for v in vector]
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    return diagnostics


def _worker(args: tuple) -> dict:
    source, target, bases_json = args
    bases = {arm: np.asarray(matrix) for arm, matrix in bases_json.items()}
    return convert_episode(pathlib.Path(source), pathlib.Path(target), bases)


def _already_converted(source: pathlib.Path, target: pathlib.Path) -> bool:
    """Whether ``target`` is a complete conversion of ``source``.

    The full pass takes hours, so an interrupted run should resume rather than
    start over. A truncated output would silently shorten an episode, so the line
    counts must match, not merely the file's existence.
    """
    if not target.exists():
        return False
    with open(source, "rb") as handle:
        expected = sum(1 for _ in handle)
    with open(target, "rb") as handle:
        actual = sum(1 for _ in handle)
    return expected == actual


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bases",
        type=pathlib.Path,
        default=pathlib.Path("data/piper_fold_cloth/estimated_bases.json"),
    )
    parser.add_argument(
        "--data-root", type=pathlib.Path, default=pathlib.Path("data/piper_fold_cloth")
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=15,
        help="Match the cgroup CPU quota, not nproc. Oversubscribing costs an "
        "order of magnitude here: 64 workers on a 15-core quota ran at 1 "
        "episode/min versus 11 at 14 workers.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Convert only the first N episodes."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reconvert episodes that already have complete output. By default "
        "they are skipped so an interrupted run can resume.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.bases.exists():
        print(
            f"error: {args.bases} not found; run script/piper_estimate_root.py first",
            file=sys.stderr,
        )
        return 1

    estimates = json.loads(args.bases.read_text())
    bases = {arm: base_transform(estimates[arm]) for arm in ARM_SLICES}
    for arm in ARM_SLICES:
        print(
            f"{arm} virtual base: {np.round(estimates[arm]['position'], 3)} "
            f"yaw {estimates[arm]['yaw_rad']:+.3f} rad "
            f"(true offset {estimates[arm]['position_error_m']:.3f} m)"
        )

    bases_json = {arm: matrix.tolist() for arm, matrix in bases.items()}

    jobs = []
    skipped = 0
    for split in ("train", "held_out"):
        source_dir = args.data_root / SOURCE_REPRESENTATION / "jsonl" / split
        target_dir = args.data_root / TARGET_REPRESENTATION / "jsonl" / split
        episodes = sorted(source_dir.glob("episode_*.jsonl"))
        if args.limit is not None:
            episodes = episodes[: args.limit]
        for path in episodes:
            target = target_dir / path.name
            if not args.overwrite and _already_converted(path, target):
                skipped += 1
                continue
            jobs.append((str(path), str(target), bases_json))

    if skipped:
        print(
            f"\nresuming: {skipped} episodes already converted, skipping "
            "(pass --overwrite to redo them)"
        )
    if not jobs:
        print("\nnothing left to convert")
        return 0
    print(f"\nconverting {len(jobs)} episodes with {args.workers} workers ...")

    totals = {
        arm: {
            "frames": 0,
            "pos_err_sum": 0.0,
            "rot_err_sum": 0.0,
            "unreached": 0,
            "branch_jumps": 0,
            "near_singular": 0,
        }
        for arm in ARM_SLICES
    }
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            for arm, values in result.items():
                for key, value in values.items():
                    totals[arm][key] += value
            if done % 100 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)} episodes", flush=True)

    scope = (
        f" (this run only; {skipped} previously converted episodes not re-scored)"
        if skipped
        else ""
    )
    print(f"\nIK conversion report{scope}")
    for arm, values in totals.items():
        frames = max(values["frames"], 1)
        print(
            f"  {arm}: {values['frames']} frames | "
            f"IK pos {values['pos_err_sum'] / frames * 1000:.3f} mm | "
            f"unreached {values['unreached'] / frames * 100:.2f}% | "
            f"branch flips {values['branch_jumps'] / frames * 100:.2f}% | "
            f"near-singular {values['near_singular'] / frames * 100:.2f}%"
        )

    worst = max(
        values["branch_jumps"] / max(values["frames"], 1) for values in totals.values()
    )
    if worst > 0.05:
        print(
            f"\nWARNING: branch flips at {worst * 100:.1f}% exceed the 5% gate. "
            "The joint stream is discontinuous, so training S5 on it would measure "
            "a broken conversion rather than the representation. Tighten "
            "piper.ik_track (lower branch_tol_rad, more restarts) first."
        )

    # The index caches would otherwise pin stale frame counts.
    for split in ("train", "held_out"):
        cache = (
            args.data_root
            / TARGET_REPRESENTATION
            / "jsonl"
            / split
            / "index_cache.json"
        )
        if cache.exists():
            cache.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
