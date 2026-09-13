#!/usr/bin/env python3
"""Prepare a shared episode split and RoboTwin S0--S4 JSONL, without copying videos."""

import argparse
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
import multiprocessing as mp
import time
import hashlib
import json
from pathlib import Path

import numpy as np

from opendm.data import se3
from opendm.data.episode_frame import (
    sample_episode_frame,
    transform_eef_states,
    atomic_json,
    write_episode,
)
from opendm.kinematics.robotwin2 import RoboTwinKinematics, ROBOTWIN_ROOT

SOURCE = Path("/kpfs_ssd/data/wzy/data/Dexmal/robotwin2-full/robotwin2.0")
REPS = {
    "s0": "joint",
    "s1": "eef_local",
    "s2": "eef_local",
    "s2_pair": "eef_local_pair",
    "s3a": "eef_unified",
    "s3": "eef_unified_pair",
    "s4": "eef_gravity_random",
}


def build_states(state, kin, representation):
    if representation == "joint":
        return state.copy()
    if representation not in REPS.values():
        raise ValueError(f"Unknown representation: {representation}")
    unified = representation not in ("eef_local", "eef_local_pair")
    left = kin.fk(state[:, :6], 0, unified)
    right = kin.fk(state[:, 7:13], 1, unified)
    blocks = []
    for pose, gripper in ((left, state[:, 6:7]), (right, state[:, 13:14])):
        blocks.extend(
            [
                pose[:, :3, 3],
                se3.unwrap_rotvec_sequence(se3.mat_to_rotvec(pose[:, :3, :3])),
                gripper,
            ]
        )
    if representation == "eef_local_pair":
        # Preserve both local state blocks. Pair only after expressing both
        # end effectors in the SAME frame; raw local poses have different bases.
        pair_left = kin.fk(state[:, :6], 0, unified=True)
        pair_right = kin.fk(state[:, 7:13], 1, unified=True)
        blocks.append(
            se3.transform_to_pos_rot6d(se3.relative_transform(pair_left, pair_right))
        )
    if representation == "eef_unified_pair":
        blocks.append(se3.transform_to_pos_rot6d(se3.relative_transform(left, right)))
    return np.concatenate(blocks, axis=-1)


@lru_cache(maxsize=1)
def _kinematics(assets):
    # Config/URDF are immutable for one invocation (recorded in manifest).
    return RoboTwinKinematics(assets)


def convert(job):
    source, destination, episode, split, reps, assets, seed, xy_range = job
    content = (Path(source) / "jsonl" / episode).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    done_path = Path(destination) / "reports/episodes" / (episode + ".json")
    previous = json.loads(done_path.read_text()) if done_path.exists() else {}
    if previous and previous["source_sha256"] != digest:
        raise ValueError(f"{episode}: source changed; use a new output root")
    pending = [
        rep
        for rep in reps
        if rep not in previous.get("representations", [])
        or not (Path(destination) / rep / "jsonl" / split / episode).exists()
    ]
    if not pending:
        return previous["frames"], False
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    if not rows or any("action" in row for row in rows):
        raise ValueError(
            f"{episode}: expected nonempty future-state supervision without explicit action"
        )
    state = np.asarray([row["state"] for row in rows], dtype=np.float64)
    if state.shape != (len(rows), 14) or not np.isfinite(state).all():
        raise ValueError(f"{episode}: invalid state")
    kin = _kinematics(assets) if any(rep != "joint" for rep in pending) else None
    for rep in pending:
        converted = build_states(state, kin, rep)
        if rep == "eef_gravity_random":
            converted = transform_eef_states(
                converted, sample_episode_frame(episode, seed, xy_range)
            )
        write_episode(
            Path(destination) / rep / "jsonl" / split / episode, rows, converted
        )
    atomic_json(
        done_path,
        {
            "source_sha256": digest,
            "frames": len(rows),
            "representations": sorted(
                set(previous.get("representations", []) + list(reps))
            ),
        },
    )
    return len(rows), True


def bounded_results(pool, function, jobs, max_pending, heartbeat_seconds=10):
    """Keep a small submission window; consume whichever job finishes first.

    Python 3.10 ProcessPoolExecutor.map eagerly submits every job. Its wakeup
    pipe can fill while submit holds _shutdown_lock, preventing the management
    thread from draining that same pipe. Bounding pending work avoids this
    deadlock as well as head-of-line blocking behind one slow episode.
    """
    iterator = iter(jobs)
    pending = {}

    def refill():
        while len(pending) < max_pending:
            job = next(iterator, None)
            if job is None:
                break
            pending[pool.submit(function, job)] = (job, time.monotonic())

    refill()
    while pending:
        completed, _ = wait(
            pending, timeout=heartbeat_seconds, return_when=FIRST_COMPLETED
        )
        if not completed:
            job, started = min(pending.values(), key=lambda item: item[1])
            print(
                f"Waiting: {len(pending)} jobs in flight; oldest {job[2]} "
                f"({time.monotonic() - started:.0f}s since submission)",
                flush=True,
            )
        for future in completed:
            job, _ = pending.pop(future)
            try:
                yield job, future.result()
            except Exception:
                print(f"Failed episode: {job[2]}", flush=True)
                for remaining in pending:
                    remaining.cancel()
                raise
        refill()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument(
        "--out-root", type=Path, default=SOURCE.parent / "robotwin2_ablation"
    )
    p.add_argument("--assets", type=Path, default=ROBOTWIN_ROOT / "assets")
    p.add_argument("--rung", nargs="+", choices=REPS, default=list(REPS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--held-out-fraction", type=float, default=0.05)
    p.add_argument("--s4-xy-range", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--limit",
        type=int,
        help="Convert first N episodes AFTER creating the full split; use a separate smoke root",
    )
    args = p.parse_args(argv)
    if (
        not 0 < args.held_out_fraction < 1
        or args.workers < 1
        or (args.limit is not None and args.limit < 1)
    ):
        p.error("Invalid fraction, workers, or limit")
    sample_episode_frame("validate", args.seed, args.s4_xy_range)
    print(
        f"Preparing {args.out_root} from {args.source} with seed {args.seed}",
        flush=True,
    )
    if args.out_root.resolve().is_relative_to(args.source.resolve()):
        p.error("Output must be outside the source tree")
    episodes = sorted(
        str(path.relative_to(args.source / "jsonl"))
        for path in (args.source / "jsonl").rglob("*.jsonl")
    )
    if not episodes:
        p.error("No source JSONL episodes found")
    print(f"Found {len(episodes)} source episodes in {args.source}/jsonl", flush=True)
    strata = defaultdict(list)
    for episode in episodes:
        parts = Path(episode).parts
        if len(parts) != 3 or parts[1] not in ("clean", "randomized"):
            raise ValueError(f"Expected task/setting/episode.jsonl: {episode}")
        strata[parts[:2]].append(episode)
    print(
        f"Found {len(strata)} task/setting strata; splitting {len(episodes)} episodes",
        flush=True,
    )
    split = {
        "seed": args.seed,
        "held_out_fraction": args.held_out_fraction,
        "train": [],
        "held_out": [],
    }
    print("Splitting into train and held-out sets", flush=True)
    for key, items in sorted(strata.items()):
        # Content-independent order; no reliance on process-specific Python hash.
        ordered = sorted(
            items, key=lambda e: hashlib.sha256(f"{args.seed}:{e}".encode()).digest()
        )
        n = min(
            len(items) - 1,
            max(1, int(np.floor(len(items) * args.held_out_fraction + 0.5))),
        )
        split["held_out"].extend(ordered[:n])
        split["train"].extend(ordered[n:])
    for name in ("train", "held_out"):
        split[name].sort()
    manifest = {
        "version": 1,
        "source": str(args.source.resolve()),
        "seed": args.seed,
        "held_out_fraction": args.held_out_fraction,
        "s4_xy_range_m": args.s4_xy_range,
        "s4_yaw_range_rad": [-np.pi, np.pi],
        "gravity": "+z unchanged; C=identity",
        "supervision": "state[t+1:t+51], repeat final state",
        "gripper": "unchanged source scale",
        "sampling": "natural frame distribution",
        "eef": "raw move_group link; no control TCP offset",
    }
    # Record assets even on S0 if available, so adding EEF groups does not change provenance.
    if (args.assets / "embodiments/aloha-agilex/config.yml").exists():
        kin = RoboTwinKinematics(args.assets)
        manifest["kinematics"] = {
            "config": str(kin.config_path.resolve()),
            "config_sha256": hashlib.sha256(kin.config_path.read_bytes()).hexdigest(),
            "urdf_sha256": hashlib.sha256(kin.urdf_path.read_bytes()).hexdigest(),
            "joint_names": kin.names,
            "eef_links": kin.tips,
            "base_links": kin.bases,
            "root_to_bases": [t.tolist() for t in kin.root_to_bases],
        }
    elif any(r != "s0" for r in args.rung):
        p.error("EEF groups require the aloha-agilex assets")

    print(f"Writing manifest and split to {args.out_root}", flush=True)

    for filename, value in [("manifest.json", manifest), ("split.json", split)]:
        path = args.out_root / filename
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(
                f"{path}: configuration/source inventory changed; use a new output root"
            )
        atomic_json(path, value)

    print("Preparing fixed S4 episode frames", flush=True)

    if "s4" in args.rung:
        atomic_json(
            args.out_root / "episode_frames.json",
            {
                e: sample_episode_frame(e, args.seed, args.s4_xy_range).tolist()
                for e in episodes
            },
        )
    reps = sorted({REPS[r] for r in args.rung})
    held_out = set(split["held_out"])
    # Remove indexes before any mutation, including resumed/interrupted conversions.
    for rep in reps:
        for name in ("train", "held_out"):
            (args.out_root / rep / "jsonl" / name / "index_cache.json").unlink(
                missing_ok=True
            )
    selected = episodes[: args.limit] if args.limit else episodes
    jobs = [
        (
            str(args.source),
            str(args.out_root),
            e,
            "held_out" if e in held_out else "train",
            reps,
            str(args.assets),
            args.seed,
            args.s4_xy_range,
        )
        for e in selected
    ]
    frames = rewritten = 0
    started = last_report = time.monotonic()
    print(
        f"Converting {len(jobs)} episodes with {args.workers} workers "
        f"(spawn, at most {args.workers * 2} in flight)",
        flush=True,
    )
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, (job, (count, changed)) in enumerate(
            bounded_results(pool, convert, jobs, args.workers * 2), 1
        ):
            frames += count
            rewritten += changed
            now = time.monotonic()
            if i == 1 or i % 100 == 0 or i == len(jobs) or now - last_report >= 10:
                elapsed = now - started
                rate = i / max(elapsed, 1e-6)
                status = {
                    "completed": i,
                    "total": len(jobs),
                    "source_episodes": len(episodes),
                    "limited": len(jobs) < len(episodes),
                    "written": rewritten,
                    "skipped": i - rewritten,
                    "frames": frames,
                    "elapsed_seconds": round(elapsed, 1),
                    "episodes_per_second": round(rate, 2),
                    "last_episode": job[2],
                    "complete": i == len(jobs),
                }
                atomic_json(args.out_root / "reports/conversion_progress.json", status)
                print(
                    f"{i}/{len(jobs)} episodes; written={rewritten}, skipped={i-rewritten}; "
                    f"{frames} frames; {rate:.1f} episodes/s; "
                    f"ETA {(len(jobs)-i)/max(rate,1e-6)/60:.1f} min; last={job[2]}",
                    flush=True,
                )
                last_report = now
    print(
        f"Full split: {len(split['train'])} train / {len(held_out)} held out; "
        f"converted/verified {len(jobs)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
