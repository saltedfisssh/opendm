#!/usr/bin/env python3
"""Derive Piper S4 from the existing S3a split, preserving videos and gripper units."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from opendm.data.episode_frame import sample_episode_frame, transform_eef_states
from opendm.data.episode_frame import atomic_json, write_episode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data/piper_fold_cloth"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--s4-xy-range", type=float, default=0.5)
    args = p.parse_args()
    sample_episode_frame("validate", args.seed, args.s4_xy_range)
    root = args.data_root
    split = json.loads((root / "split.json").read_text())
    output = root / "eef_gravity_random"
    config = {
        "version": 1,
        "seed": args.seed,
        "xy_range_m": args.s4_xy_range,
        "yaw_range_rad": [-np.pi, np.pi],
        "C": np.eye(3).tolist(),
        "gravity": "+z up, identical to existing S3a",
        "split_sha256": hashlib.sha256((root / "split.json").read_bytes()).hexdigest(),
    }
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("S4 configuration changed; use a new data root and norm cache")
    # Check the complete input inventory before writing anything.
    jobs = []
    for name in ("train", "held_out"):
        for index in split[name]:
            episode = f"episode_{index:06d}.jsonl"
            source = root / "eef_unified/jsonl" / name / episode
            if not source.is_file():
                raise FileNotFoundError(source)
            jobs.append((name, episode, source))
    atomic_json(manifest, config)
    for name in ("train", "held_out"):
        (output / "jsonl" / name / "index_cache.json").unlink(missing_ok=True)
    frames_path = output / "episode_frames.json"
    frames = {
        episode: sample_episode_frame(episode, args.seed, args.s4_xy_range).tolist()
        for _, episode, _ in jobs
    }
    atomic_json(frames_path, frames)
    for i, (name, episode, source) in enumerate(jobs, 1):
        dest = output / "jsonl" / name / episode
        receipt = output / "reports" / (episode + ".json")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if receipt.exists():
            if json.loads(receipt.read_text())["source_sha256"] != digest:
                raise ValueError(
                    f"Source changed: {source}; use a new root and norm cache"
                )
            if dest.exists():
                continue
        rows = [
            json.loads(line) for line in source.read_text().splitlines() if line.strip()
        ]
        if not rows or any("action" in row for row in rows):
            raise ValueError(
                "Expected nonempty S3a future-state data without explicit actions"
            )
        states = np.asarray([row["state"] for row in rows])
        if states.shape != (len(rows), 14) or not np.isfinite(states).all():
            raise ValueError(f"Invalid S3a states: {source}")
        write_episode(
            dest, rows, transform_eef_states(states, np.asarray(frames[episode]))
        )
        atomic_json(receipt, {"source_sha256": digest, "frames": len(rows)})
        if i % 100 == 0 or i == len(jobs):
            print(f"{i}/{len(jobs)} episodes", flush=True)


if __name__ == "__main__":
    main()
