#!/usr/bin/env python3
"""Score Piper ablation checkpoints on held-out episodes in one comparable space.

Cross-rung loss values are not comparable -- each rung predicts in different
units. This decodes every rung's prediction to the same physical quantity (each
arm's absolute flange pose in its own base frame, plus gripper width) and reports
millimetres and degrees at several points inside the action chunk, so long-horizon
drift is visible rather than averaged away.

Usage::

    python script/piper_eval_offline.py \\
        --checkpoint user_checkpoints/piper_s0/checkpoint-60000 \\
        --dataset-name piper_fold_s0 \\
        --samples 512

    # Compare several rungs in one table.
    python script/piper_eval_offline.py --compare results/*.json
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import orjson
import torch
from transformers import AutoProcessor

from opendm.constants.robot import ActionMode
from opendm.data.augmentations import NoAugmentationPipeline
from opendm.data.collator import TrainingCollator
from opendm.data.dataset import JsonlDataset, _read_jsonl_lines
from opendm.data.normalize import load_norm_stats_file
from opendm.data.transforms import (
    BuildAction,
    ChatTokenization,
    Denormalize,
    LoadImages,
    Normalize,
    PadAction,
    Pipeline,
    PixelTransform,
)
from opendm.dataset.piper_dual import held_out_dir
from opendm.dataset.register import CONVERSATION_DATA
from opendm.eval.piper_common_space import (
    RUNG_DECODERS,
    decode_to_common_space,
    ground_truth_common_space,
    pose_metrics,
)
from opendm.model.dm05.dm05_arch import DM05ForConditionalGeneration

# Representation directory per rung, needed to locate the held-out split.
RUNG_REPRESENTATIONS = {
    "piper_fold_s0": "joint",
    "piper_fold_s1": "eef_local",
    "piper_fold_s2": "eef_local",
    "piper_fold_s3": "eef_unified_pair",
    "piper_fold_s3a": "eef_unified",
    "piper_fold_s5": "joint_estimated_base",
}

RELATIVE_MODES = {
    name: spec["relative"] for name, spec in RUNG_DECODERS.items()
}


def build_eval_dataset(
    dataset_name: str,
    processor,
    action_horizon: int,
    norm_stats_path: str,
    max_length: int = 1024,
):
    """Build the held-out dataset with the training transforms, minus augmentation."""
    info = {
        key: value.value if hasattr(value, "value") else value
        for key, value in CONVERSATION_DATA[dataset_name].items()
    }
    pipeline = Pipeline(
        [
            BuildAction(
                action_horizon=action_horizon,
                action_mode=ActionMode.RELATIVE,
                relative_mode=RELATIVE_MODES[dataset_name],
            ),
            LoadImages(image_keys=info["image_keys"], image_dir=info["image_dir"]),
            # Deterministic: augmentation would add variance across rungs.
            PixelTransform(transform_pipeline=NoAugmentationPipeline()),
            Normalize(
                norm_stats_path=norm_stats_path,
                norm_keys=["state", "action"],
                use_quantiles=True,
            ),
            ChatTokenization(
                processor=processor,
                n_bins=256,
                max_length=max_length,
                image_prompts=info["image_prompts"],
                add_state=True,
            ),
            PadAction(32),
        ]
    )
    meta = {
        key: value
        for key, value in info.items()
        if key not in ("jsonl_dir", "image_dir", "image_keys", "image_prompts")
    }
    dataset = JsonlDataset(
        jsonl_dir=held_out_dir(RUNG_REPRESENTATIONS[dataset_name]),
        transforms=pipeline,
        dataset_name=dataset_name,
        dataset_meta=meta,
    )
    collator = TrainingCollator(
        pad_token_id=processor.tokenizer.pad_token_id, max_length=max_length
    )
    return dataset, collator, meta


def read_absolute_states(
    dataset: JsonlDataset, sample_index: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read the current and future absolute states for one dataset index.

    ``ChatTokenization`` returns a fresh dict and drops ``raw_lines``, ``state``
    and ``meta_data``, so the reference cannot be recovered from a transformed
    sample. Reading straight from the episode also means a bug in a rung's
    decoder cannot cancel itself out between prediction and ground truth.

    Args:
        dataset: The held-out dataset, used only for its index.
        sample_index: Index into ``dataset``.
        horizon: Action chunk length.

    Returns:
        ``(state, future_states)`` of shapes ``(D,)`` and ``(horizon, D)``.
    """
    file_index, frame_index = dataset.sample_index[sample_index]
    lines = _read_jsonl_lines(dataset.id_to_jsonl[file_index])

    state = np.asarray(orjson.loads(lines[frame_index])["state"], dtype=np.float64)

    future = []
    last = None
    terminal = len(lines) - 1
    for step in range(horizon):
        index = frame_index + 1 + step
        if index <= terminal:
            last = np.asarray(orjson.loads(lines[index])["state"], dtype=np.float64)
        future.append(last)
    return state, np.stack(future)


def evaluate(
    checkpoint: pathlib.Path,
    dataset_name: str,
    num_samples: int,
    batch_size: int,
    diffusion_steps: int,
    seed: int,
) -> dict:
    """Run the model over held-out samples and return the common-space report."""
    processor = AutoProcessor.from_pretrained(str(checkpoint))
    model = DM05ForConditionalGeneration.from_pretrained(
        str(checkpoint), dtype=torch.bfloat16
    )
    model = model.to("cuda").eval()
    action_horizon = int(model.model.config.chunk_size)
    action_dim = int(model.model.config.action_dim)

    norm_stats_path = checkpoint / "norm_stats.json"
    if not norm_stats_path.exists():
        raise FileNotFoundError(
            f"{norm_stats_path} is missing; training copies it into each checkpoint"
        )

    dataset, collator, meta = build_eval_dataset(
        dataset_name, processor, action_horizon, str(norm_stats_path)
    )
    state_desc = meta["state_desc"]

    norm_stats_file = load_norm_stats_file(str(norm_stats_path))
    action_stats = norm_stats_file.select(
        meta.get("robot_type"), meta.get("control_mode")
    )["action"]
    action_width = len(np.asarray(action_stats.mean))
    # Reuse the repo's inverse rather than re-deriving the quantile mapping.
    denormalizer = Denormalize(
        norm_stats_path=str(norm_stats_path),
        norm_keys=["action"],
        use_quantiles=True,
        norm_stats_file=norm_stats_file,
    )

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(dataset), size=min(num_samples, len(dataset)), replace=False
    )

    predicted_poses, reference_poses = [], []
    predicted_grippers, reference_grippers = [], []

    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        samples = [dataset[int(i)] for i in chunk]

        # The collator pads prompts to a common length; building the batch by
        # hand would fail as soon as two samples tokenize to different lengths.
        collated = collator(samples)
        batch = {
            key: collated[key].to("cuda")
            for key in (
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "pixel_values",
            )
        }

        mask = torch.zeros(
            len(samples), 1, action_dim, device="cuda", dtype=torch.bfloat16
        )
        mask[..., :action_width] = 1.0

        with torch.no_grad():
            actions = model.inference_action(
                **batch, diffusion_steps=diffusion_steps, action_mask=mask
            )
        actions = actions.to(torch.float32).cpu().numpy()[:, :, :action_width]

        for sample_index, prediction in zip(chunk, actions, strict=True):
            # Undo the normalization applied by the eval pipeline.
            denormalized = _denormalize(prediction, denormalizer, meta)
            raw_state, future = read_absolute_states(
                dataset, int(sample_index), action_horizon
            )

            poses, grippers = decode_to_common_space(
                dataset_name, raw_state, denormalized, state_desc
            )
            truth_poses, truth_grippers = ground_truth_common_space(
                future, state_desc, unified=RUNG_DECODERS[dataset_name]["unified"]
            )

            predicted_poses.append(poses)
            reference_poses.append(truth_poses)
            predicted_grippers.append(grippers)
            reference_grippers.append(truth_grippers)

        print(
            f"  {min(start + batch_size, len(indices))}/{len(indices)} samples",
            flush=True,
        )

    report = pose_metrics(
        np.stack(predicted_poses),
        np.stack(reference_poses),
        np.stack(predicted_grippers),
        np.stack(reference_grippers),
    )
    return {
        "checkpoint": str(checkpoint),
        "dataset_name": dataset_name,
        "num_samples": len(indices),
        "action_horizon": action_horizon,
        "action_width": action_width,
        "diffusion_steps": diffusion_steps,
        "metrics": report,
    }


def _denormalize(action: np.ndarray, denormalizer, meta: dict) -> np.ndarray:
    """Invert the training pipeline's normalization for one action chunk.

    Delegates to the repo's :class:`Denormalize`, which is the exact inverse of
    the :class:`Normalize` used to build the batch -- including the ``1e-6`` guard
    in the denominator and the rule that dimensions with ``q01 == q99 == 0`` stay
    at zero. Re-deriving it here would risk drifting from that.
    """
    return np.asarray(
        denormalizer({"action": action, "meta_data": meta})["action"], dtype=np.float64
    )


def print_table(reports: list[dict]) -> None:
    """Print one comparable table across rungs."""
    steps = ["k1", "k10", "k25", "k50"]
    header = f"{'rung':<16}{'dim':>5}"
    for step in steps:
        header += f"{step + ' mm':>11}{step + ' deg':>10}"
    header += f"{'grip mm':>10}"
    print("\n" + header)
    print("-" * len(header))
    for report in reports:
        metrics = report["metrics"]
        row = f"{report['dataset_name']:<16}{report['action_width']:>5}"
        for step in steps:
            if step in metrics:
                row += f"{metrics[step]['position_mm_mean']:>11.2f}"
                row += f"{metrics[step]['rotation_deg_mean']:>10.2f}"
            else:
                row += f"{'-':>11}{'-':>10}"
        row += f"{metrics['all_steps']['gripper_mm_mean']:>10.2f}"
        print(row)
    print(
        "\nFlange pose error in each arm's own base frame. Lower is better. "
        "Columns are horizons inside one action chunk."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--dataset-name", choices=sorted(RUNG_DECODERS))
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument(
        "--compare",
        nargs="+",
        type=pathlib.Path,
        default=None,
        help="Print a table from previously written report JSON files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.compare:
        print_table([json.loads(path.read_text()) for path in args.compare])
        return 0

    if args.checkpoint is None or args.dataset_name is None:
        print(
            "error: --checkpoint and --dataset-name are required "
            "unless --compare is used",
            file=sys.stderr,
        )
        return 1

    report = evaluate(
        args.checkpoint,
        args.dataset_name,
        args.samples,
        args.batch_size,
        args.diffusion_steps,
        args.seed,
    )
    print_table([report])

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
