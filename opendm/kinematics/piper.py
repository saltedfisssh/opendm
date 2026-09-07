"""Forward and inverse kinematics for the AgileX Piper arm.

The AgileX SDK vendored under ``third_party/pyAgxArm`` ships a modified-DH
table and a scalar forward-kinematics routine, but no Jacobian and no inverse
kinematics (IK runs on the arm firmware). This module adds the pieces needed to
convert offline between joint space and end-effector space:

* :func:`fk` — batched forward kinematics, numerically identical to the SDK's
  ``fk_from_mdh`` but vectorised over frames.
* :func:`jacobian` — the analytic spatial Jacobian of that chain.
* :func:`ik` / :func:`ik_track` — damped-least-squares inverse kinematics, with
  a sequential warm-started variant for whole trajectories.

Everything is in SI units: metres and radians. Poses are ``4x4`` homogeneous
transforms in the arm's **own base frame**, at the **flange** (the MDH chain
terminates there and carries no tool offset), matching the ``observation.qpos_ee``
field of the Piper teleoperation datasets.
"""

import numpy as np
from pyAgxArm.api.constants import (
    ROBOT_JOINT_LIMIT_PRESET_RAD,
    ROBOT_MDH_PRESET,
)

from opendm.data.se3 import make_transform, mat_to_rotvec

# The dataset's ``observation.qpos_ee`` was verified to be exactly
# ``fk_from_mdh(get_mdh("piper"), joints)``, so the plain ``piper`` table is the
# right one -- not ``piper_h`` / ``piper_l`` / ``piper_x``.
PIPER_MODEL = "piper"

# MDH parameters as ``(d, a, alpha, theta_offset)`` per link, from the SDK.
PIPER_MDH = np.asarray(ROBOT_MDH_PRESET[PIPER_MODEL], dtype=np.float64)

NUM_JOINTS = PIPER_MDH.shape[0]

# Recorded joint angles overshoot the SDK's ``piper`` limit preset on every axis
# except j1: j2 reaches -0.035 (preset min 0.0), j3 reaches +0.030 (preset max
# 0.0), and j5/j6 reach 1.341/3.001 (preset +-1.222/+-2.094). The firmware in use
# is closer to the ``piper_h`` preset. Clipping IK solutions to the narrow
# ``piper`` preset would reject configurations the real arm demonstrably holds,
# so use the elementwise union of both presets plus a margin.
_LIMIT_MARGIN_RAD = 0.05


def _joint_limits() -> np.ndarray:
    """Return ``(NUM_JOINTS, 2)`` lower/upper joint limits used for IK."""
    limits = []
    for i in range(NUM_JOINTS):
        key = f"joint{i + 1}"
        lo = min(
            ROBOT_JOINT_LIMIT_PRESET_RAD["piper"][key][0],
            ROBOT_JOINT_LIMIT_PRESET_RAD["piper_h"][key][0],
        )
        hi = max(
            ROBOT_JOINT_LIMIT_PRESET_RAD["piper"][key][1],
            ROBOT_JOINT_LIMIT_PRESET_RAD["piper_h"][key][1],
        )
        limits.append([lo - _LIMIT_MARGIN_RAD, hi + _LIMIT_MARGIN_RAD])
    return np.asarray(limits, dtype=np.float64)


PIPER_JOINT_LIMITS = _joint_limits()

# The two Piper arms face forward side by side, 60 cm apart, both reaching
# toward the middle. ``y`` points left in each arm's base frame, so the right
# arm's base sits at -y of the left arm's base. This maps a pose expressed in
# the right arm's base frame into the left arm's base frame, which is the
# unified frame used by the EEF experiments.
INTER_BASE_DISTANCE_M = 0.60

T_RIGHT_BASE_TO_LEFT_BASE = make_transform(
    np.array([0.0, -INTER_BASE_DISTANCE_M, 0.0]),
    np.eye(3),
)


def _link_transforms(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build per-link ``A`` and ``B`` factors for a batch of joint vectors.

    Each modified-DH link factors as ``A_i @ B_i`` with
    ``A_i = Rx(alpha_i) @ Tx(a_i)`` (fixed) and ``B_i = Rz(theta_i) @ Tz(d_i)``
    (joint dependent). Splitting them this way exposes the joint axis, which
    :func:`jacobian` needs: joint ``i`` rotates about the z axis of
    ``T_{i-1} @ A_i``.

    Args:
        joints: ``(N, NUM_JOINTS)`` joint angles in radians.

    Returns:
        ``(A, B)`` with shapes ``(N, NUM_JOINTS, 4, 4)``.
    """
    joints = np.atleast_2d(np.asarray(joints, dtype=np.float64))
    n = joints.shape[0]
    d, a, alpha, theta_offset = PIPER_MDH.T
    theta = joints + theta_offset

    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    mat_a = np.zeros((n, NUM_JOINTS, 4, 4))
    mat_a[:, :, 0, 0] = 1.0
    mat_a[:, :, 0, 3] = a
    mat_a[:, :, 1, 1] = ca
    mat_a[:, :, 1, 2] = -sa
    mat_a[:, :, 2, 1] = sa
    mat_a[:, :, 2, 2] = ca
    mat_a[:, :, 3, 3] = 1.0

    mat_b = np.zeros((n, NUM_JOINTS, 4, 4))
    mat_b[:, :, 0, 0] = ct
    mat_b[:, :, 0, 1] = -st
    mat_b[:, :, 1, 0] = st
    mat_b[:, :, 1, 1] = ct
    mat_b[:, :, 2, 2] = 1.0
    mat_b[:, :, 2, 3] = d
    mat_b[:, :, 3, 3] = 1.0

    return mat_a, mat_b


def fk(joints: np.ndarray) -> np.ndarray:
    """Forward kinematics to the flange pose.

    Args:
        joints: ``(..., NUM_JOINTS)`` joint angles in radians.

    Returns:
        ``(..., 4, 4)`` flange pose in the arm base frame, metres.
    """
    joints = np.asarray(joints, dtype=np.float64)
    batch = joints.shape[:-1]
    mat_a, mat_b = _link_transforms(joints.reshape(-1, NUM_JOINTS))

    acc = np.broadcast_to(np.eye(4), (mat_a.shape[0], 4, 4)).copy()
    for i in range(NUM_JOINTS):
        acc = acc @ mat_a[:, i] @ mat_b[:, i]
    return acc.reshape(*batch, 4, 4)


def jacobian(joints: np.ndarray) -> np.ndarray:
    """Analytic spatial Jacobian of :func:`fk`.

    The returned Jacobian maps joint velocities to ``[linear; angular]``
    velocity of the flange, both expressed in the base frame. The angular rows
    pair with a rotation error of ``mat_to_rotvec(R_target @ R_current.T)``.

    Args:
        joints: ``(..., NUM_JOINTS)`` joint angles in radians.

    Returns:
        ``(..., 6, NUM_JOINTS)`` Jacobian.
    """
    joints = np.asarray(joints, dtype=np.float64)
    batch = joints.shape[:-1]
    mat_a, mat_b = _link_transforms(joints.reshape(-1, NUM_JOINTS))
    n = mat_a.shape[0]

    # Frame in which joint i's rotation axis is the local z axis.
    axis_frames = np.empty((n, NUM_JOINTS, 4, 4))
    acc = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    for i in range(NUM_JOINTS):
        acc = acc @ mat_a[:, i]
        axis_frames[:, i] = acc
        acc = acc @ mat_b[:, i]
    flange_pos = acc[:, :3, 3]

    axes = axis_frames[:, :, :3, 2]
    origins = axis_frames[:, :, :3, 3]
    lever = flange_pos[:, None, :] - origins

    jac = np.empty((n, 6, NUM_JOINTS))
    jac[:, :3, :] = np.cross(axes, lever).transpose(0, 2, 1)
    jac[:, 3:, :] = axes.transpose(0, 2, 1)
    return jac.reshape(*batch, 6, NUM_JOINTS)


def pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the ``(..., 6)`` ``[translation; rotation]`` error, base frame.

    The rotation part is the rotation vector of ``R_target @ R_current.T``,
    which is the spatial (base-frame) error matching :func:`jacobian`.
    """
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    trans = target[..., :3, 3] - current[..., :3, 3]
    rot = mat_to_rotvec(
        target[..., :3, :3] @ np.swapaxes(current[..., :3, :3], -1, -2)
    )
    return np.concatenate([trans, rot], axis=-1)


def min_singular_value(joints: np.ndarray) -> np.ndarray:
    """Return the smallest singular value of the Jacobian, per configuration.

    This is the conditioning of the joint-space-to-pose map, and it matters in
    both directions. Near a wrist singularity (``j5`` close to zero, where the
    ``j4`` and ``j6`` axes align) it drops by three orders of magnitude, so a
    sub-millimetre pose error inflates into a joint error of tens of
    milliradians. Any pipeline that converts end-effector targets back to joints
    should report this alongside the pose residual, because a small pose
    residual does not imply a small joint residual.

    Args:
        joints: ``(..., NUM_JOINTS)`` joint angles in radians.

    Returns:
        ``(...)`` smallest singular value, in mixed metres/radian units.
    """
    return np.linalg.svd(jacobian(joints), compute_uv=False)[..., -1]


def ik(
    target: np.ndarray,
    joints_init: np.ndarray,
    max_iters: int = 100,
    damping: float = 1e-4,
    max_step_rad: float = 0.3,
    pos_tol_m: float = 1e-5,
    rot_tol_rad: float = 1e-5,
    joint_limits: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Damped-least-squares inverse kinematics for a single flange pose.

    Args:
        target: ``(4, 4)`` desired flange pose in the arm base frame.
        joints_init: ``(NUM_JOINTS,)`` warm start. Solution quality depends
            strongly on this: the 6-DoF chain has multiple branches, and a
            distant seed can converge to a different elbow/wrist configuration.
        max_iters: Maximum Newton iterations.
        damping: Levenberg damping added to ``J @ J.T``; larger values trade
            convergence speed for stability near singularities.
        max_step_rad: Per-iteration clamp on each joint increment.
        pos_tol_m: Position convergence threshold in metres.
        rot_tol_rad: Rotation convergence threshold in radians.
        joint_limits: ``(NUM_JOINTS, 2)`` bounds; defaults to
            :data:`PIPER_JOINT_LIMITS`.

    Returns:
        ``(joints, pos_err_m, rot_err_rad)`` for the best iterate found.
    """
    limits = PIPER_JOINT_LIMITS if joint_limits is None else joint_limits
    joints = np.clip(
        np.asarray(joints_init, dtype=np.float64).copy(),
        limits[:, 0],
        limits[:, 1],
    )
    eye6 = np.eye(6)

    best_joints = joints.copy()
    best_cost = np.inf
    best_errs = (np.inf, np.inf)

    for _ in range(max_iters):
        err = pose_error(fk(joints), target)
        pos_err = float(np.linalg.norm(err[:3]))
        rot_err = float(np.linalg.norm(err[3:]))
        # Weight rotation in metres-equivalent so one scalar ranks iterates.
        cost = pos_err + 0.1 * rot_err
        if cost < best_cost:
            best_cost = cost
            best_joints = joints.copy()
            best_errs = (pos_err, rot_err)
        if pos_err < pos_tol_m and rot_err < rot_tol_rad:
            break

        jac = jacobian(joints)
        step = jac.T @ np.linalg.solve(jac @ jac.T + damping * eye6, err)
        joints = np.clip(
            joints + np.clip(step, -max_step_rad, max_step_rad),
            limits[:, 0],
            limits[:, 1],
        )

    return best_joints, best_errs[0], best_errs[1]


def ik_multistart(
    target: np.ndarray,
    seeds: np.ndarray,
    **ik_kwargs,
) -> tuple[np.ndarray, float, float]:
    """Solve IK from several seeds and keep the closest-reaching solution.

    Single-seed IK on this chain conflates "the pose is unreachable" with "the
    seed was in the wrong basin", which makes it useless as a reachability
    oracle -- exactly what a base-pose estimator needs. Trying a spread of seeds
    separates the two.

    Args:
        target: ``(4, 4)`` desired flange pose in the arm base frame.
        seeds: ``(S, NUM_JOINTS)`` warm starts, tried in order.
        **ik_kwargs: Forwarded to :func:`ik`.

    Returns:
        ``(joints, pos_err_m, rot_err_rad)`` of the best seed.
    """
    seeds = np.atleast_2d(np.asarray(seeds, dtype=np.float64))
    best: tuple[np.ndarray, float, float] | None = None
    for seed in seeds:
        candidate = ik(target, seed, **ik_kwargs)
        if best is None or candidate[1] < best[1]:
            best = candidate
    assert best is not None, "seeds must not be empty"
    return best


def seed_bank(
    previous: np.ndarray | None = None,
    num_random: int = 6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build a seed set for :func:`ik_multistart`.

    The previous solution goes first when available, then the neutral pose, then
    uniform samples across the joint range to cover the remaining branches.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    seeds = []
    if previous is not None:
        seeds.append(np.asarray(previous, dtype=np.float64))
    seeds.append(neutral_joints())
    lower, upper = PIPER_JOINT_LIMITS.T
    seeds.extend(rng.uniform(lower, upper) for _ in range(num_random))
    return np.stack(seeds)


def ik_track(
    targets: np.ndarray,
    joints_init: np.ndarray,
    branch_tol_rad: float = 0.35,
    restarts: int | None = None,
    retry_pos_tol_m: float = 1e-4,
    **ik_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve IK along a trajectory with sequential warm starts.

    Warm starting from the previous frame is what keeps the solution on one
    kinematic branch. A frame is retried from :func:`seed_bank` when it either
    fails to reach the target or jumps by more than ``branch_tol_rad``; among the
    candidates, ones that converge win, and among those the least-drifting wins.

    Retrying on the residual matters as much as retrying on the drift. A solve
    that stalls barely moves from its seed, so its drift is *small* -- keying
    retries on drift alone lets a stalled frame through and then hands its bad
    solution to the next frame as a warm start, which cascades. Measured on the
    Piper capture, that failure mode reported ~30-65% of frames as unreachable
    when a multi-start oracle could reach every one of them.

    Args:
        targets: ``(T, 4, 4)`` flange poses in the arm base frame.
        joints_init: ``(NUM_JOINTS,)`` seed for the first frame.
        branch_tol_rad: Maximum tolerated per-joint change between frames.
        restarts: Number of random re-seeds tried on a retry. Defaults to ``4``.
        retry_pos_tol_m: Position residual above which a frame is retried.
        **ik_kwargs: Forwarded to :func:`ik`.

    Returns:
        ``(joints, pos_err, rot_err, branch_jump)`` with shapes ``(T, NUM_JOINTS)``,
        ``(T,)``, ``(T,)`` and ``(T,)`` (boolean).
    """
    targets = np.asarray(targets, dtype=np.float64)
    horizon = targets.shape[0]
    restarts = 4 if restarts is None else restarts

    joints_out = np.empty((horizon, NUM_JOINTS))
    pos_errs = np.empty(horizon)
    rot_errs = np.empty(horizon)
    jumps = np.zeros(horizon, dtype=bool)

    rng = np.random.default_rng(0)
    previous = np.asarray(joints_init, dtype=np.float64).copy()

    for t in range(horizon):
        solution, pos_err, rot_err = ik(targets[t], previous, **ik_kwargs)
        drift = float(np.abs(solution - previous).max())

        stalled = pos_err > retry_pos_tol_m
        jumped = t > 0 and drift > branch_tol_rad
        if restarts > 0 and (stalled or jumped):
            # Rank by (did not converge, drift): a converged solution always beats
            # a stalled one, and among converged ones continuity decides.
            candidates = [(stalled, drift, pos_err, rot_err, solution)]
            for seed in seed_bank(previous, num_random=restarts, rng=rng):
                alt, alt_pos, alt_rot = ik(targets[t], seed, **ik_kwargs)
                candidates.append(
                    (
                        alt_pos > retry_pos_tol_m,
                        float(np.abs(alt - previous).max()),
                        alt_pos,
                        alt_rot,
                        alt,
                    )
                )
            _, drift, pos_err, rot_err, solution = min(
                candidates, key=lambda c: (c[0], c[1])
            )

        jumps[t] = t > 0 and drift > branch_tol_rad

        joints_out[t] = solution
        pos_errs[t] = pos_err
        rot_errs[t] = rot_err
        previous = solution

    return joints_out, pos_errs, rot_errs, jumps


def neutral_joints() -> np.ndarray:
    """Return a mid-range seed pose, used when no warm start is available."""
    return np.array([0.0, 0.9, -0.9, 0.0, 0.5, 0.0], dtype=np.float64)
