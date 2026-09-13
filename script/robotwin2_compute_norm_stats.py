#!/usr/bin/env python3
"""Compute separate train-only statistics with exactly the training action encoding."""

import argparse
from pathlib import Path
from opendm.exp.dm05_exp import DM05DataConfig
from opendm.dataset.robotwin2_ablation import (
    DATA_ROOT,
    SOURCE,
    REPRESENTATIONS,
    RELATIVE_MODES,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rung", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS)
    )
    p.add_argument("--data-root", type=Path, default=Path(DATA_ROOT))
    p.add_argument("--source", type=Path, default=Path(SOURCE))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--action-horizon", type=int, default=50)
    p.add_argument("--max-batches", type=int)
    args = p.parse_args()
    for rung in args.rung:
        config = DM05DataConfig(
            dataset_name=f"robotwin2_ablation_{rung}",
            relative_mode=RELATIVE_MODES[rung],
            jsonl_dir=str(args.data_root / REPRESENTATIONS[rung] / "jsonl/train"),
            image_dir=str(args.source / "video"),
            norm_stats_root=str(args.data_root / "norm_stats"),
            compute_norm_stats_max_batches=args.max_batches,
        )
        path = config.norm_stats_path(args.action_horizon)
        if not path.exists():
            config.compute_norm_stats(
                args.action_horizon,
                batch_size=args.batch_size,
                num_workers=args.workers,
            )
        print(f"{rung}: {path}", flush=True)


if __name__ == "__main__":
    main()
