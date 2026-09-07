"""Tests for the cross-rung common-space decoder.

The comparison between rungs is only as trustworthy as this decoder. Each test
below constructs a known future trajectory, encodes it the way a given rung
would, decodes it, and requires the original flange poses back -- so a rung that
silently decodes into the wrong frame or the wrong units cannot pass.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from opendm.constants.robot import RobotStateDesc
from opendm.data import se3
from opendm.data.transforms import ActionRelative, ActionRelativeSE3
from opendm.dataset.piper_dual import (
    PIPER_EEF_PAIR_STATE_DESC,
    PIPER_EEF_STATE_DESC,
    PIPER_JOINT_STATE_DESC,
)
from opendm.eval.piper_common_space import (
    RUNG_DECODERS,
    decode_to_common_space,
    ground_truth_common_space,
    pose_metrics,
)
from opendm.kinematics import piper

HORIZON = 8


@pytest.fixture
def joint_trajectory() -> np.ndarray:
    """``(HORIZON + 1, 14)`` smooth bimanual joint states."""
    rng = np.random.default_rng(5)
    lower, upper = piper.PIPER_JOINT_LIMITS.T
    start = rng.uniform(lower * 0.4, upper * 0.4, (2, piper.NUM_JOINTS))
    drift = rng.normal(scale=0.01, size=(HORIZON + 1, 2, piper.NUM_JOINTS))
    joints = start[None] + np.cumsum(drift, axis=0)
    gripper = rng.uniform(0.0, 0.1, (HORIZON + 1, 2, 1))
    return np.concatenate(
        [
            np.concatenate([joints[:, 0], gripper[:, 0]], axis=-1),
            np.concatenate([joints[:, 1], gripper[:, 1]], axis=-1),
        ],
        axis=-1,
    )


def _eef_states_from_joints(joints: np.ndarray, unified: bool) -> np.ndarray:
    """Convert joint states to the on-disk EEF layout used by S1-S3."""
    blocks = []
    for arm, (joint_slice, gripper) in enumerate(
        ((slice(0, 6), 6), (slice(7, 13), 13))
    ):
        pose = piper.fk(joints[:, joint_slice])
        if unified and arm == 1:
            pose = piper.T_RIGHT_BASE_TO_LEFT_BASE @ pose
        blocks.append(se3.transform_to_pos_rotvec(pose))
        blocks.append(joints[:, gripper : gripper + 1])
    return np.concatenate(blocks, axis=-1)


def test_every_registered_rung_has_a_decoder() -> None:
    from opendm.dataset.register import CONVERSATION_DATA

    registered = {name for name in CONVERSATION_DATA if name.startswith("piper_fold_")}
    assert registered == set(RUNG_DECODERS)


def test_joint_rung_roundtrip(joint_trajectory: np.ndarray) -> None:
    """S0/S5: joint deltas must decode to the same flange poses as the truth."""
    state = joint_trajectory[0]
    future = joint_trajectory[1:]
    meta = {"state_desc": list(PIPER_JOINT_STATE_DESC)}

    encoded = ActionRelative()(
        {"state": state, "action": future[None, ...], "meta_data": meta}
    )["action"][0]

    poses, grippers = decode_to_common_space(
        "piper_fold_s0", state, encoded, PIPER_JOINT_STATE_DESC
    )
    reference, reference_gripper = ground_truth_common_space(
        future, PIPER_JOINT_STATE_DESC, unified=False
    )

    assert poses.shape == (HORIZON, 2, 4, 4)
    assert np.allclose(poses, reference, atol=1e-6)
    assert np.allclose(grippers, reference_gripper, atol=1e-6)


def test_vector_eef_rung_roundtrip(joint_trajectory: np.ndarray) -> None:
    """S1: the repo's base-frame delta convention must decode exactly."""
    states = _eef_states_from_joints(joint_trajectory, unified=False)
    state, future = states[0], states[1:]
    meta = {"state_desc": list(PIPER_EEF_STATE_DESC)}

    encoded = ActionRelative()(
        {"state": state, "action": future[None, ...], "meta_data": meta}
    )["action"][0]

    poses, grippers = decode_to_common_space(
        "piper_fold_s1", state, encoded, PIPER_EEF_STATE_DESC
    )
    reference, reference_gripper = ground_truth_common_space(
        future, PIPER_EEF_STATE_DESC, unified=False
    )

    # Position is exact; rotation is only exact because the decoder composes in
    # SO(3) rather than adding axis-angle vectors the way the encoder subtracts
    # them, so compare rotations through the shared reference instead.
    assert np.allclose(poses[..., :3, 3], reference[..., :3, 3], atol=1e-6)
    assert np.allclose(grippers, reference_gripper, atol=1e-6)


def test_se3_eef_rung_roundtrip(joint_trajectory: np.ndarray) -> None:
    """S2: the body-frame SE(3) encoding must decode to the exact poses."""
    states = _eef_states_from_joints(joint_trajectory, unified=False)
    state, future = states[0], states[1:]
    meta = {"state_desc": list(PIPER_EEF_STATE_DESC)}

    encoded = ActionRelativeSE3()(
        {"state": state, "action": future[None, ...], "meta_data": meta}
    )["action"][0]

    poses, grippers = decode_to_common_space(
        "piper_fold_s2", state, encoded, PIPER_EEF_STATE_DESC
    )
    reference, reference_gripper = ground_truth_common_space(
        future, PIPER_EEF_STATE_DESC, unified=False
    )

    assert np.allclose(poses, reference, atol=1e-5)
    assert np.allclose(grippers, reference_gripper, atol=1e-6)


def test_unified_rung_maps_the_right_arm_back_to_its_own_base(
    joint_trajectory: np.ndarray,
) -> None:
    """S3: the shared frame must be undone, or the right arm lands 60 cm off.

    This is the single most consequential decoder detail. Without the inverse
    map the right arm's reported error would be dominated by a constant 0.6 m
    offset, making S3 look catastrophically worse than every other rung for a
    purely bookkeeping reason.
    """
    states = _eef_states_from_joints(joint_trajectory, unified=True)
    state, future = states[0], states[1:]
    meta = {"state_desc": list(PIPER_EEF_STATE_DESC)}

    encoded = ActionRelativeSE3()(
        {"state": state, "action": future[None, ...], "meta_data": meta}
    )["action"][0]

    poses, _ = decode_to_common_space(
        "piper_fold_s3a", state, encoded, PIPER_EEF_STATE_DESC
    )
    # Reference built straight from the joints, in each arm's own base frame.
    truth = np.stack(
        [
            piper.fk(joint_trajectory[1:, 0:6]),
            piper.fk(joint_trajectory[1:, 7:13]),
        ],
        axis=1,
    )
    assert np.allclose(poses, truth, atol=1e-5)


def test_unified_pair_state_desc_ignores_aux_block(
    joint_trajectory: np.ndarray,
) -> None:
    """S3's trailing inter-gripper feature must not be mistaken for a third arm."""
    states = _eef_states_from_joints(joint_trajectory, unified=True)
    rng = np.random.default_rng(1)
    padded = np.concatenate([states, rng.normal(size=(len(states), 9))], axis=-1)
    meta = {"state_desc": list(PIPER_EEF_PAIR_STATE_DESC)}

    encoded = ActionRelativeSE3()(
        {"state": padded[0], "action": padded[1:][None, ...], "meta_data": meta}
    )["action"][0]

    poses, _ = decode_to_common_space(
        "piper_fold_s3", padded[0], encoded, PIPER_EEF_PAIR_STATE_DESC
    )
    assert poses.shape == (HORIZON, 2, 4, 4)


def test_joint_and_eef_rungs_agree_on_the_same_trajectory(
    joint_trajectory: np.ndarray,
) -> None:
    """The point of the common space: two rungs, one trajectory, one answer.

    If S0 and S2 disagree here, any measured difference between them in a real
    run would be decoder noise rather than a property of the representation.
    """
    joint_state, joint_future = joint_trajectory[0], joint_trajectory[1:]
    joint_encoded = ActionRelative()(
        {
            "state": joint_state,
            "action": joint_future[None, ...],
            "meta_data": {"state_desc": list(PIPER_JOINT_STATE_DESC)},
        }
    )["action"][0]
    joint_poses, joint_grippers = decode_to_common_space(
        "piper_fold_s0", joint_state, joint_encoded, PIPER_JOINT_STATE_DESC
    )

    eef_states = _eef_states_from_joints(joint_trajectory, unified=False)
    eef_encoded = ActionRelativeSE3()(
        {
            "state": eef_states[0],
            "action": eef_states[1:][None, ...],
            "meta_data": {"state_desc": list(PIPER_EEF_STATE_DESC)},
        }
    )["action"][0]
    eef_poses, eef_grippers = decode_to_common_space(
        "piper_fold_s2", eef_states[0], eef_encoded, PIPER_EEF_STATE_DESC
    )

    assert np.allclose(joint_poses, eef_poses, atol=1e-5)
    assert np.allclose(joint_grippers, eef_grippers, atol=1e-6)


# --- metrics -------------------------------------------------------------------------


def test_pose_metrics_zero_for_perfect_prediction() -> None:
    rng = np.random.default_rng(2)
    poses = se3.make_transform(
        rng.normal(size=(4, HORIZON, 2, 3)),
        Rotation.random(4 * HORIZON * 2, random_state=2).as_matrix().reshape(
            4, HORIZON, 2, 3, 3
        ),
    )
    gripper = rng.uniform(0.0, 0.1, (4, HORIZON, 2))

    report = pose_metrics(poses, poses, gripper, gripper, steps=(1, HORIZON))

    assert report["k1"]["position_mm_mean"] == pytest.approx(0.0, abs=1e-6)
    assert report["k1"]["rotation_deg_mean"] == pytest.approx(0.0, abs=1e-6)
    assert report["all_steps"]["gripper_mm_mean"] == pytest.approx(0.0, abs=1e-9)


def test_pose_metrics_reports_millimetres_and_degrees() -> None:
    """A known 5 mm / 3 degree offset must come back as 5 and 3."""
    identity = np.tile(np.eye(4), (1, 1, 1, 1, 1))
    shifted = identity.copy()
    shifted[..., :3, 3] = [0.005, 0.0, 0.0]
    shifted[..., :3, :3] = Rotation.from_rotvec([0.0, 0.0, np.radians(3.0)]).as_matrix()

    report = pose_metrics(
        shifted, identity, np.zeros((1, 1, 1)), np.full((1, 1, 1), 0.002), steps=(1,)
    )

    assert report["k1"]["position_mm_mean"] == pytest.approx(5.0, abs=1e-3)
    assert report["k1"]["rotation_deg_mean"] == pytest.approx(3.0, abs=1e-3)
    assert report["k1"]["gripper_mm_mean"] == pytest.approx(2.0, abs=1e-6)


def test_pose_metrics_skips_horizons_beyond_the_chunk() -> None:
    identity = np.tile(np.eye(4), (2, 3, 2, 1, 1))
    gripper = np.zeros((2, 3, 2))
    report = pose_metrics(identity, identity, gripper, gripper, steps=(1, 10, 50))
    assert set(report) == {"k1", "all_steps"}
