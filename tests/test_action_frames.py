"""Tests for the end-effector action representations.

These lock down the two properties the UMI-style representation is chosen for:

* :class:`ActionRelativeSE3` and :class:`ActionAbsoluteSE3` are exact inverses,
  so a policy trained on relative targets can be decoded at serving time; and
* relative targets are invariant to the choice of world frame, which is what
  makes an arbitrary SLAM origin and an uncalibrated robot base drop out.

The second property is why experiment S4 ("randomise the world frame") is a test
rather than a training run: a frame-invariant pipeline produces identical targets
under any world frame, so training on randomised frames cannot differ.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from opendm.constants.robot import ActionMode, RobotStateDesc
from opendm.data import se3
from opendm.data.transforms import (
    ABSOLUTE_SE3_DIMS_PER_ARM,
    RELATIVE_SE3_DIMS_PER_ARM,
    ActionAbsoluteSE3,
    ActionRelativeSE3,
    BuildAction,
    _eef_arm_blocks,
)

HORIZON = 12
NUM_ARMS = 2

EEF_STATE_DESC = (
    [RobotStateDesc.EEF] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.EEF] * 6
    + [RobotStateDesc.GRIPPER]
)
EEF_PAIR_STATE_DESC = EEF_STATE_DESC + [RobotStateDesc.AUX] * 9


def _random_arm_state(rng: np.random.Generator, num_frames: int) -> np.ndarray:
    """Build ``(num_frames, 14)`` plausible bimanual EEF states."""
    frames = []
    for _ in range(NUM_ARMS):
        position = rng.uniform([-0.1, -0.5, 0.1], [0.5, 0.2, 0.7], (num_frames, 3))
        seed = int(rng.integers(1 << 30))
        rotvec = Rotation.random(num_frames, random_state=seed).as_rotvec()
        gripper = rng.uniform(0.0, 0.1, (num_frames, 1))
        frames.append(np.concatenate([position, rotvec, gripper], axis=-1))
    return np.concatenate(frames, axis=-1)


@pytest.fixture
def sample() -> dict:
    rng = np.random.default_rng(7)
    frames = _random_arm_state(rng, HORIZON + 1)
    return {
        "state": frames[0],
        "action": frames[1:][None, ...],
        "meta_data": {"state_desc": list(EEF_STATE_DESC)},
    }


# --- descriptor parsing --------------------------------------------------------------


def test_eef_arm_blocks_finds_both_arms() -> None:
    blocks = _eef_arm_blocks(EEF_STATE_DESC)
    assert blocks == [(slice(0, 6), 6), (slice(7, 13), 13)]


def test_eef_arm_blocks_ignores_aux_tail() -> None:
    expected = [(slice(0, 6), 6), (slice(7, 13), 13)]
    assert _eef_arm_blocks(EEF_PAIR_STATE_DESC) == expected


def test_eef_arm_blocks_empty_for_joint_layout() -> None:
    joint_desc = [RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER]
    assert _eef_arm_blocks(joint_desc) == []


def test_eef_arm_blocks_rejects_run_that_is_not_a_multiple_of_six() -> None:
    with pytest.raises(ValueError, match="not a multiple of 6"):
        _eef_arm_blocks([RobotStateDesc.EEF] * 7 + [RobotStateDesc.GRIPPER])


def test_eef_arm_blocks_requires_a_trailing_gripper() -> None:
    with pytest.raises(ValueError, match="not followed by a gripper"):
        _eef_arm_blocks([RobotStateDesc.EEF] * 6 + [RobotStateDesc.JOINT])


# --- encode / decode -----------------------------------------------------------------


def test_relative_se3_shape_and_gripper_passthrough(sample: dict) -> None:
    original = sample["action"].copy()
    encoded = ActionRelativeSE3()(dict(sample))

    assert encoded["action"].shape == (1, HORIZON, NUM_ARMS * RELATIVE_SE3_DIMS_PER_ARM)
    assert encoded["action"].dtype == np.float32
    assert encoded["action_mask"].all()
    # Gripper commands stay absolute, at the tail of each arm block.
    for arm, gripper_dim in enumerate((6, 13)):
        offset = arm * RELATIVE_SE3_DIMS_PER_ARM + 9
        assert np.allclose(
            encoded["action"][..., offset], original[..., gripper_dim], atol=1e-6
        )


def test_relative_se3_is_zero_translation_when_target_equals_state(
    sample: dict,
) -> None:
    """A target equal to the current pose must encode as the identity transform."""
    held = dict(sample)
    held["action"] = np.repeat(sample["state"][None, None, :], HORIZON, axis=1)

    encoded = ActionRelativeSE3()(held)["action"]

    for arm in range(NUM_ARMS):
        block = encoded[..., arm * RELATIVE_SE3_DIMS_PER_ARM :][..., :9]
        identity = se3.transform_to_pos_rot6d(np.eye(4))
        assert np.allclose(block, identity, atol=1e-5)


def test_absolute_se3_inverts_relative_se3(sample: dict) -> None:
    expected = sample["action"].copy()
    encoded = ActionRelativeSE3()(dict(sample))

    decoded = ActionAbsoluteSE3()(
        {
            "state": sample["state"],
            "action": encoded["action"],
            "meta_data": sample["meta_data"],
        }
    )["action"]

    assert decoded.shape == (1, HORIZON, NUM_ARMS * ABSOLUTE_SE3_DIMS_PER_ARM)
    # Compare as rigid transforms: the rotation chart may differ by a 2*pi turn.
    for arm, eef_slice in enumerate((slice(0, 6), slice(7, 13))):
        base = arm * ABSOLUTE_SE3_DIMS_PER_ARM
        out = slice(base, base + 6)
        got = se3.pos_rotvec_to_transform(decoded[..., out])
        want = se3.pos_rotvec_to_transform(expected[..., eef_slice])
        assert np.allclose(got, want, atol=1e-5)


def test_absolute_se3_recovers_gripper(sample: dict) -> None:
    expected = sample["action"].copy()
    encoded = ActionRelativeSE3()(dict(sample))
    decoded = ActionAbsoluteSE3()(
        {
            "state": sample["state"],
            "action": encoded["action"],
            "meta_data": sample["meta_data"],
        }
    )["action"]

    for arm, gripper_dim in enumerate((6, 13)):
        offset = arm * ABSOLUTE_SE3_DIMS_PER_ARM + 6
        assert np.allclose(
            decoded[..., offset], expected[..., gripper_dim], atol=1e-6
        )


def test_relative_se3_requires_an_eef_block() -> None:
    with pytest.raises(ValueError, match="at least one EEF arm block"):
        ActionRelativeSE3()(
            {
                "state": np.zeros(7),
                "action": np.zeros((1, 4, 7)),
                "meta_data": {
                    "state_desc": [RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER]
                },
            }
        )


def test_aux_dims_are_dropped_from_action_targets() -> None:
    """The inter-gripper feature is an observation, never a prediction target."""
    rng = np.random.default_rng(9)
    frames = _random_arm_state(rng, HORIZON + 1)
    padded = np.concatenate([frames, rng.normal(size=(HORIZON + 1, 9))], axis=-1)

    encoded = ActionRelativeSE3()(
        {
            "state": padded[0],
            "action": padded[1:][None, ...],
            "meta_data": {"state_desc": list(EEF_PAIR_STATE_DESC)},
        }
    )["action"]

    assert encoded.shape[-1] == NUM_ARMS * RELATIVE_SE3_DIMS_PER_ARM


# --- the property S4 tests -----------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_relative_se3_targets_are_world_frame_invariant(seed: int) -> None:
    """Re-expressing every pose in a random world frame must not move the targets.

    This is the mathematical content of experiment S4: because the anchor and the
    targets are transformed together, ``T_anchor^-1 @ T_target`` is unchanged.
    Randomising the world frame during training therefore cannot produce
    different action targets, so it is asserted here instead of trained.
    """
    rng = np.random.default_rng(seed)
    frames = _random_arm_state(rng, HORIZON + 1)
    meta = {"state_desc": list(EEF_STATE_DESC)}

    baseline = ActionRelativeSE3()(
        {"state": frames[0], "action": frames[1:][None, ...], "meta_data": meta}
    )["action"]

    world = se3.make_transform(
        rng.uniform(-5.0, 5.0, 3),
        Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix(),
    )
    shifted = frames.copy()
    for eef_slice in (slice(0, 6), slice(7, 13)):
        poses = world @ se3.pos_rotvec_to_transform(frames[..., eef_slice])
        shifted[..., eef_slice] = se3.transform_to_pos_rotvec(poses)

    rebased = ActionRelativeSE3()(
        {"state": shifted[0], "action": shifted[1:][None, ...], "meta_data": meta}
    )["action"]

    assert np.allclose(rebased, baseline, atol=1e-5)


def test_vector_relative_targets_are_not_world_frame_invariant() -> None:
    """The contrast that motivates the SE(3) encoder.

    Elementwise subtraction of axis-angle values is a base-frame delta. It is
    invariant to a pure translation of the world frame but not to a rotation of
    it, so it cannot absorb an arbitrary world origin the way the body-frame form
    does.
    """
    rng = np.random.default_rng(3)
    frames = _random_arm_state(rng, HORIZON + 1)
    meta = {"state_desc": list(EEF_STATE_DESC)}

    from opendm.data.transforms import ActionRelative

    baseline = ActionRelative()(
        {"state": frames[0], "action": frames[1:][None, ...], "meta_data": meta}
    )["action"]

    world = se3.make_transform(
        np.zeros(3), Rotation.from_rotvec([0.0, 0.0, 0.9]).as_matrix()
    )
    shifted = frames.copy()
    for eef_slice in (slice(0, 6), slice(7, 13)):
        poses = world @ se3.pos_rotvec_to_transform(frames[..., eef_slice])
        shifted[..., eef_slice] = se3.transform_to_pos_rotvec(poses)

    rebased = ActionRelative()(
        {"state": shifted[0], "action": shifted[1:][None, ...], "meta_data": meta}
    )["action"]

    assert not np.allclose(rebased, baseline, atol=1e-3)


# --- BuildAction dispatch ------------------------------------------------------------


def test_build_action_rejects_unknown_relative_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported relative_mode"):
        BuildAction(action_horizon=4, relative_mode="quaternion")


def test_build_action_str_distinguishes_relative_modes() -> None:
    """The norm-stats filename is derived from this string, so the modes must differ."""
    vector = str(BuildAction(action_horizon=4, relative_mode="vector"))
    body = str(BuildAction(action_horizon=4, relative_mode="se3"))
    assert vector != body
    assert "ActionRelativeSE3" in body


def test_build_action_absolute_mode_ignores_relative_mode() -> None:
    absolute = str(
        BuildAction(
            action_horizon=4, action_mode=ActionMode.ABSOLUTE, relative_mode="se3"
        )
    )
    assert "ActionRelativeSE3" not in absolute
