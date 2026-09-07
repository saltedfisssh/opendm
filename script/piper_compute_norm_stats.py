#!/usr/bin/env python3
"""Precompute normalization statistics for the Piper ablation rungs.

Training computes these lazily on rank 0 while every other rank polls, which
wastes an eight-GPU allocation for the ~15 minutes it takes per rung. Running it
here first means each training job starts immediately.

Statistics are keyed by dataset name and action transform
(``DM05DataConfig.norm_stats_path``), so each rung gets its own file and they
cannot collide.

Usage::

    python script/piper_compute_norm_stats.py
    python script/piper_compute_norm_stats.py --rung s2 s3 --workers 64
"""

import argparse
import sys
import time

from opendm.constants.robot import ActionMode
from opendm.exp.dm05_exp import DM05DataConfig

# Which relative encoding each rung is trained with. This must match the
# ``--data-config.relative-mode`` used at training time, or the statistics will
# be computed for a different action space than the one the model sees.
RUNG_RELATIVE_MODES = {
    "s0": "vector",
    "s1": "vector",
    "s2": "se3",
    "s3": "se3",
    "s3a": "se3",
    "s5": "vector",
}

NORM_STATS_ROOT = "./norm_stats/piper"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--rung",
        nargs="+",
        default=["s0", "s1", "s2", "s3", "s3a"],
        help="Rung suffixes to process; s5 needs its dataset built first.",
    )
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap batches for a smoke run; omit to use the whole split.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    for rung in args.rung:
        if rung not in RUNG_RELATIVE_MODES:
            print(f"error: unknown rung {rung!r}", file=sys.stderr)
            return 1

        config = DM05DataConfig(
            dataset_name=f"piper_fold_{rung}",
            action_mode=ActionMode.RELATIVE,
            relative_mode=RUNG_RELATIVE_MODES[rung],
            norm_stats_root=NORM_STATS_ROOT,
            compute_norm_stats_max_batches=args.max_batches,
        )
        path = config.norm_stats_path(args.action_horizon)
        if path.exists():
            print(f"{rung}: already present at {path}, skipping")
            continue

        started = time.time()
        config.compute_norm_stats(
            args.action_horizon,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        print(f"{rung}: {path} in {(time.time() - started) / 60:.1f} min", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
