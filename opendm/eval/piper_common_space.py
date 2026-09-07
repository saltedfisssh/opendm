"""Decode each ablation rung's action space into one comparable space.

Cross-rung MSE is meaningless: S0 predicts joint deltas in radians, S1 predicts
base-frame pose deltas in metres and axis-angle, S2 and S3 predict body-frame
SE(3) transforms with a 6D rotation. Reporting raw loss per rung would compare
different units.

Everything is therefore decoded to the same physical quantity: the **absolute
flange pose of each arm, in that arm's own base frame**, plus gripper width.
Joint-space rungs get there through forward kinematics; end-effector rungs are
already poses and only need the relative encoding undone (and, for unified-frame
rungs, the right arm mapped back out of the shared frame).
"""

import numpy as np

from opendm.constants.robot import RobotStateDesc
from opendm.data import se3
from opendm.data.transforms import (
    RELATIVE_SE3_DIMS_PER_ARM,
    _eef_arm_blocks,
    _state_desc_value,
)
from opendm.kinematics import piper

# How each rung encodes actions, and whether its poses live in a shared frame.
RUNG_DECODERS = {
    "piper_fold_s0": {"space": "joint", "relative": "vector", "unified": False},
    "piper_fold_s1": {"space": "eef", "relative": "vector", "unified": False},
    "piper_fold_s2": {"space": "eef", "relative": "se3", "unified": False},
    "piper_fold_s3": {"space": "eef", "relative": "se3", "unified": True},
    "piper_fold_s3a": {"space": "eef", "relative": "se3", "unified": True},
    "piper_fold_s5": {"space": "joint", "relative": "vector", "unified": False},
}

# Arms are ordered left then right, matching the state layout.
ARM_NAMES = ("left", "right")


def _joint_blocks(state_desc) -> list[tuple[slice, int]]:
    """Locate ``6 x JOINT + 1 x GRIPPER`` arm blocks in a state descriptor."""
    values = [_state_desc_value(desc) for desc in state_desc]
    joint_id = _state_desc_value(RobotStateDesc.JOINT)
    gripper_id = _state_desc_value(RobotStateDesc.GRIPPER)

    blocks = []
    index = 0
    while index < len(values):
        if values[index] != joint_id:
            index += 1
            continue
        start = index
        while index < len(values) and values[index] == joint_id:
            index += 1
        for offset in range(start, index, 6):
            gripper = offset + 6
            if gripper < len(values) and values[gripper] == gripper_id:
                blocks.append((slice(offset, offset + 6), gripper))
        index = max(index, blocks[-1][1] + 1) if blocks else index
    return blocks


def decode_joint_rung(
    state: np.ndarray, action: np.ndarray, state_desc
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a joint-space rung to per-arm flange poses in each arm's base frame.

    Args:
        state: ``(D,)`` current absolute joint state.
        action: ``(H, D)`` predicted deltas, with gripper dims already absolute.
        state_desc: Per-dimension descriptors.

    Returns:
        ``(poses, grippers)`` of shapes ``(H, num_arms, 4, 4)`` and ``(H, num_arms)``.
    """
    blocks = _joint_blocks(state_desc)
    absolute = state[None, :] + action
    poses, grippers = [], []
    for joint_slice, gripper in blocks:
        poses.append(piper.fk(absolute[:, joint_slice]))
        grippers.append(action[:, gripper])
    return np.stack(poses, axis=1), np.stack(grippers, axis=1)


def decode_eef_rung(
    state: np.ndarray,
    action: np.ndarray,
    state_desc,
    relative: str,
    unified: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode an end-effector rung to per-arm flange poses in each arm's base frame.

    Args:
        state: ``(D,)`` current absolute state, ``[xyz, axis-angle, gripper]`` per arm.
        action: ``(H, A)`` predicted action. ``A`` is ``D`` for the vector encoding
            and ``10 * num_arms`` for the body-frame SE(3) encoding.
        state_desc: Per-dimension descriptors.
        relative: ``"vector"`` or ``"se3"``.
        unified: Whether poses are expressed in the left arm's base frame, in which
            case the right arm is mapped back into its own base frame.

    Returns:
        ``(poses, grippers)`` of shapes ``(H, num_arms, 4, 4)`` and ``(H, num_arms)``.
    """
    blocks = _eef_arm_blocks(state_desc)
    poses, grippers = [], []

    for arm_index, (eef_slice, gripper) in enumerate(blocks):
        anchor = se3.pos_rotvec_to_transform(state[eef_slice])

        if relative == "se3":
            offset = arm_index * RELATIVE_SE3_DIMS_PER_ARM
            delta = se3.pos_rot6d_to_transform(action[:, offset : offset + 9])
            absolute = se3.absolute_transform(anchor, delta)
            gripper_values = action[:, offset + 9]
        else:
            # The repo's base-frame convention: translations add, and rotations
            # compose in quaternion space rather than adding axis-angle vectors.
            position = state[eef_slice][:3] + action[:, eef_slice][:, :3]
            rotation = (
                se3.rotvec_to_mat(action[:, eef_slice][:, 3:6]) @ anchor[:3, :3]
            )
            absolute = se3.make_transform(position, rotation)
            gripper_values = action[:, gripper]

        if unified and ARM_NAMES[arm_index] == "right":
            absolute = se3.transform_inverse(piper.T_RIGHT_BASE_TO_LEFT_BASE) @ absolute

        poses.append(absolute)
        grippers.append(gripper_values)

    return np.stack(poses, axis=1), np.stack(grippers, axis=1)


def decode_to_common_space(
    dataset_name: str,
    state: np.ndarray,
    action: np.ndarray,
    state_desc,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the decoder for ``dataset_name``.

    Returns:
        ``(poses, grippers)``: flange poses per arm in that arm's own base frame,
        shape ``(H, num_arms, 4, 4)``, and gripper widths in metres, ``(H, num_arms)``.
    """
    if dataset_name not in RUNG_DECODERS:
        raise KeyError(
            f"no decoder registered for {dataset_name!r}; "
            f"known rungs: {sorted(RUNG_DECODERS)}"
        )
    spec = RUNG_DECODERS[dataset_name]
    if spec["space"] == "joint":
        return decode_joint_rung(state, action, state_desc)
    return decode_eef_rung(
        state, action, state_desc, spec["relative"], spec["unified"]
    )


def ground_truth_common_space(
    future_states: np.ndarray, state_desc, unified: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Build reference poses straight from the held-out absolute states.

    Read from the episode rather than by inverting a predicted encoding, so a bug
    in a decoder cannot cancel itself out between prediction and reference.

    Args:
        future_states: ``(H, D)`` absolute states of the future frames.
        state_desc: Per-dimension descriptors.
        unified: Whether end-effector poses are in the shared left-base frame.

    Returns:
        ``(poses, grippers)`` in the same layout as :func:`decode_to_common_space`.
    """
    joint_blocks = _joint_blocks(state_desc)
    if joint_blocks:
        poses, grippers = [], []
        for joint_slice, gripper in joint_blocks:
            poses.append(piper.fk(future_states[:, joint_slice]))
            grippers.append(future_states[:, gripper])
        return np.stack(poses, axis=1), np.stack(grippers, axis=1)

    poses, grippers = [], []
    for arm_index, (eef_slice, gripper) in enumerate(_eef_arm_blocks(state_desc)):
        pose = se3.pos_rotvec_to_transform(future_states[:, eef_slice])
        if unified and ARM_NAMES[arm_index] == "right":
            pose = se3.transform_inverse(piper.T_RIGHT_BASE_TO_LEFT_BASE) @ pose
        poses.append(pose)
        grippers.append(future_states[:, gripper])
    return np.stack(poses, axis=1), np.stack(grippers, axis=1)


def pose_metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    predicted_gripper: np.ndarray,
    reference_gripper: np.ndarray,
    steps: tuple[int, ...] = (1, 10, 25, 50),
) -> dict:
    """Position, rotation and gripper error at selected horizons.

    Args:
        predicted: ``(N, H, num_arms, 4, 4)`` predicted flange poses.
        reference: ``(N, H, num_arms, 4, 4)`` reference flange poses.
        predicted_gripper: ``(N, H, num_arms)`` predicted widths in metres.
        reference_gripper: ``(N, H, num_arms)`` reference widths in metres.
        steps: One-based horizons to report.

    Returns:
        Nested dict keyed by ``f"k{step}"`` then metric name. Units are
        millimetres and degrees, chosen so the numbers are readable on a robot.
    """
    error = piper.pose_error(predicted, reference)
    position_mm = np.linalg.norm(error[..., :3], axis=-1) * 1000.0
    rotation_deg = np.degrees(np.linalg.norm(error[..., 3:], axis=-1))
    gripper_mm = np.abs(predicted_gripper - reference_gripper) * 1000.0

    horizon = predicted.shape[1]
    report = {}
    for step in steps:
        if step > horizon:
            continue
        index = step - 1
        report[f"k{step}"] = {
            "position_mm_mean": float(position_mm[:, index].mean()),
            "position_mm_p95": float(np.percentile(position_mm[:, index], 95)),
            "rotation_deg_mean": float(rotation_deg[:, index].mean()),
            "rotation_deg_p95": float(np.percentile(rotation_deg[:, index], 95)),
            "gripper_mm_mean": float(gripper_mm[:, index].mean()),
        }
    report["all_steps"] = {
        "position_mm_mean": float(position_mm.mean()),
        "rotation_deg_mean": float(rotation_deg.mean()),
        "gripper_mm_mean": float(gripper_mm.mean()),
    }
    return report
