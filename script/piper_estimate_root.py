#!/usr/bin/env python3
"""Estimate a virtual robot base from end-effector trajectories alone (rung S5).

Premise: a UMI-style capture gives end-effector poses in some world frame but no
joint angles and no idea where the robot base is. To train a joint-space policy
on such data you must invent a base, then run IK. This script does that, and --
because this dataset does have ground truth -- reports how far off the invented
base is.

What the geometry actually allows
--------------------------------
Joint 1 is a pure rotation about the base z axis, so the flange's cylindrical
radius and height in the base frame do not depend on it. With a gravity-aligned
world frame (which UMI captures get from an IMU) that makes reachability a
yaw-invariant 2D condition on the base *position* only.

Measured on this capture, that condition is satisfied *exactly* by a set of base
positions 0.7-0.9 m across in every axis (779 grid points at 5 cm spacing with
6100 frames of workspace coverage, 1708 with only 320 frames): the demonstrated
motion occupies a small region well inside the arm's workspace, so hard
reachability cannot pin the base down. More coverage tightens the set but nowhere
near a unique answer. Objectives built on IK residual or reachable-fraction are
therefore not estimators here -- a sweep over the true offset is flatly
non-monotone, and a joint-limit-violation term is identically zero because IK
clips to limits.

So this does not pretend to identify the true base. It selects, from the feasible
set, the base that yields the best-*conditioned* joint trajectories:

* joints far from their limits,
* Jacobians far from singular, and
* smooth motion frame to frame.

That is the right criterion for the downstream purpose. A policy is trained and
deployed with the same invented base, so a constant offset from the truth is an
invertible reparameterisation the network absorbs. What actually hurts is joints
pinned against limits, IK branch flips, and near-singular configurations where a
sub-millimetre pose error becomes a large joint error.

Usage::

    python script/piper_estimate_root.py --episodes 40
    python script/piper_estimate_root.py --report-only
"""

import argparse
import glob
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from opendm.data import se3
from opendm.kinematics import piper

# Ground truth in the unified (left-base) frame, used only for the error report.
TRUE_BASES = {
    "left": np.zeros(3),
    "right": np.array([0.0, -piper.INTER_BASE_DISTANCE_M, 0.0]),
}

ARM_STATE_SLICES = {"left": slice(0, 6), "right": slice(7, 13)}

WORKSPACE_CELL_M = 0.01


def build_workspace_mask(
    num_samples: int = 400_000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterise the q1-invariant reachable ``(radius, height)`` region.

    Returns:
        ``(mask, radius_edges, height_edges)`` for table-lookup membership tests.
    """
    rng = np.random.default_rng(seed)
    lower, upper = piper.PIPER_JOINT_LIMITS.T
    joints = rng.uniform(lower, upper, (num_samples, piper.NUM_JOINTS))
    joints[:, 0] = 0.0  # radius and height do not depend on joint 1.

    flange = piper.fk(joints)[:, :3, 3]
    radius = np.linalg.norm(flange[:, :2], axis=1)
    height = flange[:, 2]

    radius_edges = np.arange(0.0, radius.max() + 2 * WORKSPACE_CELL_M, WORKSPACE_CELL_M)
    height_edges = np.arange(
        height.min() - 2 * WORKSPACE_CELL_M,
        height.max() + 2 * WORKSPACE_CELL_M,
        WORKSPACE_CELL_M,
    )
    counts, _, _ = np.histogram2d(radius, height, bins=[radius_edges, height_edges])
    return counts > 0, radius_edges, height_edges


def unreachable_fraction(
    positions: np.ndarray,
    base: np.ndarray,
    workspace: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    """Fraction of flange positions outside the reachable region for a candidate base."""
    mask, radius_edges, height_edges = workspace
    delta = positions - base
    radius = np.linalg.norm(delta[:, :2], axis=1)
    height = delta[:, 2]
    ri = np.clip(np.searchsorted(radius_edges, radius) - 1, 0, mask.shape[0] - 1)
    hi = np.clip(np.searchsorted(height_edges, height) - 1, 0, mask.shape[1] - 1)
    return float(1.0 - mask[ri, hi].mean())


def conditioning_cost(
    segments: list[np.ndarray],
    base_xyz: np.ndarray,
    base_yaw: float,
    restarts: int = 2,
) -> dict:
    """Solve IK for a candidate base and score the resulting joint trajectories.

    Scoring runs on **contiguous** pose segments, not strided samples. Smoothness
    and branch-flip rate are frame-to-frame quantities: measured across a stride
    of 200 frames they report the arm genuinely moving, not the solver jumping
    branches, and are worthless as diagnostics.

    Args:
        segments: List of ``(T, 4, 4)`` contiguous flange-pose segments in the
            gravity-aligned world frame.
        base_xyz: Candidate base position.
        base_yaw: Candidate base yaw about the world z axis.
        restarts: Retry seeds per frame. Must stay above zero even while merely
            *scoring*: with no retries a stalled solve is counted as unreachable,
            so ``unreached_frac`` would rank candidate bases by how often the
            solver got stuck rather than by geometry. Two is enough to separate
            the two; the final chosen base is re-scored with more.

    Returns:
        Dict with the individual terms plus a scalar ``cost``.
    """
    base = se3.make_transform(
        base_xyz, se3.rotvec_to_mat(np.array([0.0, 0.0, base_yaw]))
    )
    inverse = se3.transform_inverse(base)
    lower, upper = piper.PIPER_JOINT_LIMITS.T
    span = upper - lower

    pos_errs, rot_errs, headrooms, sigmas, steps, jumps = [], [], [], [], [], []
    for segment in segments:
        joints, pos_err, rot_err, jump = piper.ik_track(
            inverse @ segment,
            piper.neutral_joints(),
            max_iters=80,
            restarts=restarts,
        )
        pos_errs.append(pos_err)
        rot_errs.append(rot_err)
        # Distance to the nearer limit as a fraction of each joint's range: 0 is
        # pinned at a limit, 0.5 is dead centre.
        headrooms.append(np.minimum(joints - lower, upper - joints) / span)
        sigmas.append(piper.min_singular_value(joints))
        steps.append(np.linalg.norm(np.diff(joints, axis=0), axis=1))
        jumps.append(jump)

    pos_err = np.concatenate(pos_errs)
    rot_err = np.concatenate(rot_errs)
    headroom = np.concatenate(headrooms)
    sigma = np.concatenate(sigmas)
    step = np.concatenate(steps)
    jump = np.concatenate(jumps)

    terms = {
        "pos_err_m": float(pos_err.mean()),
        "rot_err_rad": float(rot_err.mean()),
        "unreached_frac": float((pos_err > 2e-3).mean()),
        # A single pinned joint anywhere would zero out a min, so report how often
        # the trajectory runs close to a limit instead.
        "pinned_frac": float((headroom.min(axis=-1) < 0.01).mean()),
        "p05_headroom": float(np.percentile(headroom.min(axis=-1), 5)),
        "median_sigma": float(np.median(sigma)),
        "near_singular_frac": float((sigma < 1e-2).mean()),
        "mean_step_rad": float(step.mean()),
        "branch_jump_frac": float(jump.mean()),
    }
    # Feasibility dominates; among feasible bases prefer headroom, conditioning
    # and smoothness. Weights are scale-matched so no single term saturates.
    terms["cost"] = (
        50.0 * terms["unreached_frac"]
        + 10.0 * terms["pos_err_m"]
        + 5.0 * terms["pinned_frac"]
        + 3.0 * terms["near_singular_frac"]
        + 5.0 * terms["mean_step_rad"]
        + 20.0 * terms["branch_jump_frac"]
    )
    return terms


def load_poses(
    representation: str,
    num_episodes: int,
    segment_length: int,
    segments_per_episode: int,
    coverage_stride: int = 10,
    max_segments: int | None = None,
    seed: int = 0,
) -> tuple[dict, dict]:
    """Load pose data per arm for the two stages, which have different needs.

    Stage 1 (feasibility) is a table lookup per position, so it should see as much
    of the demonstrated workspace as possible -- otherwise the chosen base is
    feasible only for the slice it was fitted on, and the full-dataset IK pass
    later reports poses it cannot reach.

    Stage 2 (conditioning) costs one IK solve per frame, so it uses a small number
    of *contiguous* segments at native rate, which is what makes smoothness and
    branch-flip rate meaningful.

    Args:
        representation: Converted representation directory to read.
        num_episodes: Number of episodes to draw from.
        segment_length: Frames per contiguous scoring segment.
        segments_per_episode: Scoring segments sampled from each episode.
        coverage_stride: Frame stride for the stage-1 coverage sample.
        max_segments: Cap on stage-2 scoring segments. Coverage and scoring scale
            independently: coverage should be as broad as possible because it is
            free, while every extra scoring segment multiplies the IK cost of
            every candidate base.
        seed: Segment-sampling seed.

    Returns:
        ``(coverage, segments)`` where ``coverage[arm]`` is ``(N, 3)`` flange
        positions and ``segments[arm]`` is a list of ``(segment_length, 4, 4)``.
    """
    pattern = f"data/piper_fold_cloth/{representation}/jsonl/train/*.jsonl"
    files = sorted(glob.glob(pattern))[:num_episodes]
    if not files:
        raise FileNotFoundError(f"no episodes under {pattern}")

    rng = np.random.default_rng(seed)
    coverage: dict[str, list[np.ndarray]] = {arm: [] for arm in ARM_STATE_SLICES}
    segments: dict[str, list[np.ndarray]] = {arm: [] for arm in ARM_STATE_SLICES}

    for path in files:
        states = np.asarray(
            [json.loads(line)["state"] for line in open(path)], dtype=np.float64
        )
        for arm, arm_slice in ARM_STATE_SLICES.items():
            coverage[arm].append(states[::coverage_stride, arm_slice][:, :3])

        if len(states) < segment_length:
            continue
        starts = rng.integers(
            0, len(states) - segment_length, size=segments_per_episode
        )
        for start in starts:
            window = states[start : start + segment_length]
            for arm, arm_slice in ARM_STATE_SLICES.items():
                segments[arm].append(se3.pos_rotvec_to_transform(window[:, arm_slice]))

    if max_segments is not None:
        keep = rng.permutation(len(next(iter(segments.values()))))[:max_segments]
        segments = {arm: [values[i] for i in keep] for arm, values in segments.items()}

    return (
        {arm: np.concatenate(values) for arm, values in coverage.items()},
        segments,
    )


def _score_one(args: tuple) -> dict:
    segments, base_xyz, base_yaw = args
    return conditioning_cost(segments, np.asarray(base_xyz), float(base_yaw))


def _score_many(
    segments: list[np.ndarray],
    candidates: list[tuple[np.ndarray, float]],
    workers: int,
) -> list[dict]:
    """Score candidate bases in parallel.

    IK here is pure-Python at roughly 12 ms per solve, so a serial sweep over a
    few dozen candidates takes tens of minutes. The scores are independent, so a
    process pool turns that into a minute or two.
    """
    jobs = [(segments, base.tolist(), yaw) for base, yaw in candidates]
    if workers <= 1:
        return [_score_one(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_score_one, jobs))


def estimate_base(
    coverage: np.ndarray,
    segments: list[np.ndarray],
    workspace: tuple[np.ndarray, np.ndarray, np.ndarray],
    workers: int = 32,
    grid_size: int = 6,
    refine_rounds: int = 3,
    verbose: bool = True,
) -> tuple[np.ndarray, float, dict]:
    """Two-stage base estimate: feasible position set, then conditioning refinement.

    Args:
        coverage: ``(N, 3)`` broad sample of flange positions for the feasibility
            stage. Should span the whole demonstrated workspace, or the chosen
            base will be infeasible for poses it never saw.
        segments: Contiguous flange-pose segments for the IK-scoring stage.
        workspace: Output of :func:`build_workspace_mask`.
        workers: Processes used to score candidates in parallel.
        grid_size: Feasible positions sampled for the initial sweep. Each is
            scored at two yaws.
        refine_rounds: Coordinate-pattern refinement rounds after the sweep.
        verbose: Print stage progress.

    Runtime is dominated by IK: one solve per loaded frame per candidate. The
    defaults are sized to finish in a few minutes per arm; raise ``grid_size``,
    ``refine_rounds`` and the segment count for a more thorough search.
    """
    positions = np.asarray(coverage, dtype=np.float64)

    # Stage 1: enumerate the feasible position set on a coarse grid. This is
    # cheap (no IK) and, as documented above, typically large.
    axis = np.arange(-0.45, 0.46, 0.05)
    candidates = []
    for dx in axis:
        for dy in axis:
            for dz in axis:
                centre = positions.mean(axis=0) + np.array([dx, dy, dz])
                if unreachable_fraction(positions, centre, workspace) == 0.0:
                    candidates.append(centre)

    if not candidates:
        raise RuntimeError("no reachable base position found; check the world frame")
    candidates = np.asarray(candidates)
    if verbose:
        spread = candidates.max(axis=0) - candidates.min(axis=0)
        print(
            f"    stage 1: {len(candidates)} feasible base positions, "
            f"spread {np.round(spread, 3)} m"
        )

    # Stage 2: score candidates by joint-trajectory conditioning. Each score is
    # one IK solve per loaded frame, so this is the expensive stage; the scores
    # are independent, so they run in a process pool. The feasible set is far too
    # large to score exhaustively, so sample it.
    rng = np.random.default_rng(0)
    subset = candidates[
        rng.choice(len(candidates), size=min(grid_size, len(candidates)), replace=False)
    ]
    grid = [(candidate, yaw) for candidate in subset for yaw in (0.0, np.pi)]

    scores = _score_many(segments, grid, workers)
    best_index = int(np.argmin([terms["cost"] for terms in scores]))
    best = (grid[best_index][0], grid[best_index][1], scores[best_index])
    if verbose:
        print(f"    stage 2: scored {len(grid)} candidates", flush=True)

    # Local refinement. A serial optimiser such as Powell would serialise the IK
    # solves, so instead shrink a coordinate pattern around the winner and
    # evaluate each round's neighbours in parallel.
    current = np.concatenate([best[0], [best[1]]])
    current_cost = best[2]["cost"]
    step = np.array([0.08, 0.08, 0.08, 0.4])
    for _ in range(refine_rounds):
        neighbours = []
        for axis_index in range(4):
            for sign in (-1.0, 1.0):
                candidate = current.copy()
                candidate[axis_index] += sign * step[axis_index]
                neighbours.append((candidate[:3], float(candidate[3])))
        scores = _score_many(segments, neighbours, workers)
        costs = [terms["cost"] for terms in scores]
        winner = int(np.argmin(costs))
        if costs[winner] < current_cost:
            current = np.concatenate([neighbours[winner][0], [neighbours[winner][1]]])
            current_cost = costs[winner]
        else:
            step *= 0.5
        if verbose:
            print(f"    refine: cost {current_cost:.4f}", flush=True)

    # Re-score the winner with branch recovery on: this is the base that will
    # actually be used, so its diagnostics should reflect real tracking.
    terms = conditioning_cost(segments, current[:3], current[3], restarts=4)
    return current[:3], float(current[3]), terms


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--representation", default="eef_unified")
    parser.add_argument(
        "--episodes",
        type=int,
        default=8,
        help="Episodes to draw scoring segments from. Every candidate base costs "
        "one IK solve per loaded frame, and stage 2 scores 32 candidates plus a "
        "local search, so this dominates runtime (~5 min per arm at the default).",
    )

    parser.add_argument(
        "--segment-length",
        type=int,
        default=40,
        help="Frames per contiguous segment; kept at native 30 Hz so that "
        "smoothness and branch-flip rate are meaningful.",
    )
    parser.add_argument("--segments-per-episode", type=int, default=1)
    parser.add_argument(
        "--max-segments",
        type=int,
        default=12,
        help="Cap on stage-2 scoring segments. Raise --episodes freely for "
        "workspace coverage (it is a table lookup); this is what actually costs "
        "IK time, so keep it small.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Processes used to score candidate bases in parallel.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=6,
        help="Feasible base positions sampled for the initial sweep.",
    )
    parser.add_argument("--refine-rounds", type=int, default=3)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("data/piper_fold_cloth/estimated_bases.json"),
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print the report without writing the estimate to disk.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("rasterising the q1-invariant reachable region ...", flush=True)
    workspace = build_workspace_mask()
    mask = workspace[0]
    print(f"  {mask.sum()} of {mask.size} cells reachable", flush=True)

    coverage, segments = load_poses(
        args.representation,
        args.episodes,
        args.segment_length,
        args.segments_per_episode,
        max_segments=args.max_segments,
    )
    print(
        f"loaded {len(next(iter(coverage.values())))} coverage frames and "
        f"{len(next(iter(segments.values())))} scoring segments of "
        f"{args.segment_length} frames per arm from {args.episodes} episodes\n",
        flush=True,
    )

    estimates = {}
    for arm, arm_segments in segments.items():
        print(f"  {arm} arm:", flush=True)
        position, yaw, terms = estimate_base(
            coverage[arm],
            arm_segments,
            workspace,
            workers=args.workers,
            grid_size=args.grid_size,
            refine_rounds=args.refine_rounds,
            verbose=True,
        )
        error = float(np.linalg.norm(position - TRUE_BASES[arm]))
        estimates[arm] = {
            "position": position.tolist(),
            "yaw_rad": yaw,
            "true_position": TRUE_BASES[arm].tolist(),
            "position_error_m": error,
            "diagnostics": terms,
        }
        print(
            f"    stage 2: base {np.round(position, 3)} yaw {yaw:+.3f} rad\n"
            f"    error vs ground truth: {error:.3f} m\n"
            f"    IK pos {terms['pos_err_m'] * 1000:.2f} mm | "
            f"unreached {terms['unreached_frac'] * 100:.1f}% | "
            f"pinned {terms['pinned_frac'] * 100:.1f}% | "
            f"near-singular {terms['near_singular_frac'] * 100:.1f}% | "
            f"branch jumps {terms['branch_jump_frac'] * 100:.1f}%",
            flush=True,
        )

    print(
        "\nNote: the position error above is not a failure of the optimiser. The "
        "reachability condition alone admits a feasible set tens of centimetres "
        "across, so the base is not identifiable from these trajectories. The "
        "estimate is chosen for joint-trajectory conditioning, which is what "
        "matters when the same base is used for training and deployment."
    )

    if not args.report_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(estimates, indent=2) + "\n")
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
