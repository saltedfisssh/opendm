"""Regression tests for :mod:`opendm.kinematics.piper`.

Correctness is anchored on two independent references:

* the AgileX SDK's own scalar ``fk_from_mdh``, and
* the real teleoperation dataset, whose ``observation.qpos_ee`` field was
  produced by that SDK routine.

The dataset checks are skipped when the read-only capture mount is absent, so
the suite still runs on machines without it.
"""

import glob

import numpy as np
import pytest
from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh

from opendm.data import se3
from opendm.kinematics import piper

DATASET_ROOT = "/mnt/xiaoyu_teleop_data/piper/20260907/piper_fold_cloth_in_place"

# Left arm occupies state[0:7] and qpos_ee[0:8]; right arm state[7:14], qpos_ee[8:16].
ARM_SLICES = {
    "left": (slice(0, 6), slice(0, 7)),
    "right": (slice(7, 13), slice(8, 15)),
}


@pytest.fixture
def random_joints() -> np.ndarray:
    rng = np.random.default_rng(11)
    lower, upper = piper.PIPER_JOINT_LIMITS.T
    return rng.uniform(lower, upper, (256, piper.NUM_JOINTS))


def _load_dataset_frames(num_episodes: int = 4):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    files = sorted(glob.glob(f"{DATASET_ROOT}/data/chunk-000/*.parquet"))[:num_episodes]
    if not files:
        pytest.skip(f"teleop dataset not mounted at {DATASET_ROOT}")

    states, poses = [], []
    for path in files:
        table = pq.read_table(
            path, columns=["observation.state", "observation.qpos_ee"]
        ).to_pydict()
        states.append(np.asarray(table["observation.state"], dtype=np.float64))
        poses.append(np.asarray(table["observation.qpos_ee"], dtype=np.float64))
    return np.concatenate(states), np.concatenate(poses)


def test_fk_matches_sdk_reference(random_joints: np.ndarray) -> None:
    """The batched chain multiply must reproduce the SDK's scalar routine."""
    mdh = list(get_mdh(piper.PIPER_MODEL))
    reference = np.asarray([fk_from_mdh(mdh, list(q)) for q in random_joints])

    got = piper.fk(random_joints)
    assert np.allclose(got[:, :3, 3], reference[:, :3], atol=1e-12)
    assert np.allclose(got[:, :3, :3], se3.rpy_to_mat(reference[:, 3:6]), atol=1e-12)


def test_fk_preserves_batch_shape() -> None:
    joints = np.zeros((3, 5, piper.NUM_JOINTS))
    assert piper.fk(joints).shape == (3, 5, 4, 4)
    assert piper.fk(np.zeros(piper.NUM_JOINTS)).shape == (4, 4)


def test_jacobian_matches_finite_differences(random_joints: np.ndarray) -> None:
    """Guards the analytic Jacobian, which is what makes IK converge."""
    joints = random_joints[:64]
    base = piper.fk(joints)
    analytic = piper.jacobian(joints)

    step = 1e-7
    numeric = np.empty_like(analytic)
    for i in range(piper.NUM_JOINTS):
        nudged = joints.copy()
        nudged[:, i] += step
        numeric[:, :, i] = piper.pose_error(base, piper.fk(nudged)) / step

    assert np.allclose(analytic, numeric, atol=1e-6)


def test_pose_error_zero_for_identical_poses(random_joints: np.ndarray) -> None:
    poses = piper.fk(random_joints)
    assert np.allclose(piper.pose_error(poses, poses), 0.0, atol=1e-12)


def test_ik_recovers_joints_from_own_forward_kinematics(
    random_joints: np.ndarray,
) -> None:
    """Round-trip IK from a nearby seed must reproduce the target pose.

    Pose accuracy is asserted everywhere. Joint agreement is only asserted for
    well-conditioned configurations: this arm is non-redundant, so near a wrist
    singularity a matched pose still permits joint deviations of tens of
    milliradians along the ill-conditioned direction (equal and opposite on j4
    and j6). That is a property of the mechanism, not a solver defect, so the
    guard is split accordingly.
    """
    rng = np.random.default_rng(23)
    joints = random_joints[:48]
    targets = piper.fk(joints)
    conditioning = piper.min_singular_value(joints)
    seeds = np.clip(
        joints + rng.normal(scale=0.05, size=joints.shape),
        piper.PIPER_JOINT_LIMITS[:, 0],
        piper.PIPER_JOINT_LIMITS[:, 1],
    )

    well_conditioned = 0
    for target, seed, truth, sigma in zip(
        targets, seeds, joints, conditioning, strict=True
    ):
        solved, pos_err, rot_err = piper.ik(target, seed)
        assert pos_err < 1e-4, f"position residual {pos_err}"
        assert rot_err < 1e-3, f"rotation residual {rot_err}"
        if sigma > 1e-2:
            well_conditioned += 1
            assert np.abs(solved - truth).max() < 1e-2

    assert well_conditioned > len(joints) // 2, "fixture is degenerate"


def test_min_singular_value_collapses_at_wrist_singularity() -> None:
    """j5 near zero aligns the j4 and j6 axes, breaking the EEF-to-joint map."""
    singular = np.array([0.2, 1.0, -1.0, 0.3, 0.0, -0.4])
    regular = np.array([0.2, 1.0, -1.0, 0.3, 0.9, -0.4])
    assert piper.min_singular_value(singular) < 1e-3
    assert piper.min_singular_value(regular) > 1e-2


def test_min_singular_value_preserves_batch_shape() -> None:
    assert piper.min_singular_value(np.zeros((4, 3, piper.NUM_JOINTS))).shape == (4, 3)
    assert piper.min_singular_value(np.zeros(piper.NUM_JOINTS)).shape == ()


def test_ik_respects_joint_limits() -> None:
    """An unreachable target must still return a feasible configuration."""
    target = se3.make_transform(np.array([3.0, 3.0, 3.0]), np.eye(3))
    solved, _, _ = piper.ik(target, piper.neutral_joints())
    assert (solved >= piper.PIPER_JOINT_LIMITS[:, 0] - 1e-9).all()
    assert (solved <= piper.PIPER_JOINT_LIMITS[:, 1] + 1e-9).all()


def test_widened_limits_contain_sdk_presets() -> None:
    """The widened limits must never be tighter than the SDK's own ``piper`` preset."""
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD

    preset = ROBOT_JOINT_LIMIT_PRESET_RAD["piper"]
    for i in range(piper.NUM_JOINTS):
        lower, upper = preset[f"joint{i + 1}"]
        assert piper.PIPER_JOINT_LIMITS[i, 0] <= lower
        assert piper.PIPER_JOINT_LIMITS[i, 1] >= upper


def test_inter_base_transform_is_pure_lateral_translation() -> None:
    """Both arms face forward, 60 cm apart, so the offset is -y translation only."""
    transform = piper.T_RIGHT_BASE_TO_LEFT_BASE
    assert np.allclose(transform[:3, :3], np.eye(3))
    assert np.allclose(transform[:3, 3], [0.0, -piper.INTER_BASE_DISTANCE_M, 0.0])


# --- checks against the real capture -------------------------------------------------


def test_fk_reproduces_dataset_qpos_ee() -> None:
    """``observation.qpos_ee`` is this FK applied to ``observation.state``.

    Agreement is not exact on every frame: the capture PC samples joint feedback
    and end-pose feedback from separate CAN messages, so during fast motion the
    two disagree by up to ~6 mm. The median frame agrees to a few micrometres,
    which is what pins the MDH table, the frame and the quaternion order. That
    residual jitter is also why the conversion pipeline recomputes poses from
    joints instead of reading ``qpos_ee``: it keeps every representation
    describing one identical trajectory.
    """
    states, poses = _load_dataset_frames()

    for joint_slice, pose_slice in ARM_SLICES.values():
        got = piper.fk(states[:, joint_slice])
        expected = se3.pos_quat_to_transform(poses[:, pose_slice])
        errors = piper.pose_error(got, expected)
        position = np.linalg.norm(errors[:, :3], axis=-1)

        assert np.median(position) < 1e-5
        assert np.percentile(position, 90) < 1e-3
        assert position.max() < 1e-2


def test_dataset_joints_lie_within_widened_limits() -> None:
    """The recorded motion overshoots the SDK preset; the widened bounds must hold it."""
    states, _ = _load_dataset_frames()
    joints = np.concatenate([states[:, 0:6], states[:, 7:13]])
    assert (joints >= piper.PIPER_JOINT_LIMITS[:, 0]).all()
    assert (joints <= piper.PIPER_JOINT_LIMITS[:, 1]).all()


def test_ik_track_retries_on_a_stalled_solve_not_only_on_a_jump() -> None:
    """A stalled solve barely moves from its seed, so drift cannot detect it.

    Regression guard. Keying retries on drift alone let non-converged frames
    through -- their drift is small precisely *because* they did not move -- and
    each one then seeded the next frame, cascading. On the real capture that
    reported 30-65% of frames as unreachable when every one of them was in fact
    reachable. The trajectory below starts from a deliberately hostile seed so
    that the first frames stall unless the residual triggers a retry.
    """
    rng = np.random.default_rng(31)
    lower, upper = piper.PIPER_JOINT_LIMITS.T
    start = np.array([1.8, 2.4, -2.6, 1.2, -1.0, 1.7])
    joints = np.clip(
        start + np.cumsum(rng.normal(scale=0.02, size=(80, piper.NUM_JOINTS)), axis=0),
        lower,
        upper,
    )
    targets = piper.fk(joints)

    solved, pos_err, _, jumps = piper.ik_track(
        targets, piper.neutral_joints(), restarts=4
    )

    assert (pos_err <= 1e-3).all(), (
        f"{(pos_err > 1e-3).sum()} frames failed to converge; "
        "retries are not firing on the residual"
    )
    assert jumps.mean() < 0.05
    assert np.allclose(piper.fk(solved)[:, :3, 3], targets[:, :3, 3], atol=1e-3)


def test_ik_track_branch_flips_are_measured_frame_to_frame() -> None:
    """Flip rate is only meaningful at the native rate the data is written at.

    Subsampling inflates it: the arm genuinely moves further between widely
    spaced samples, and that motion is indistinguishable from a solver jump. The
    S5 dataset is written at every frame, so that is where the gate applies.
    """
    rng = np.random.default_rng(17)
    lower, upper = piper.PIPER_JOINT_LIMITS.T
    start = rng.uniform(lower * 0.3, upper * 0.3, piper.NUM_JOINTS)
    joints = np.clip(
        start + np.cumsum(rng.normal(scale=0.01, size=(120, piper.NUM_JOINTS)), axis=0),
        lower,
        upper,
    )
    targets = piper.fk(joints)

    _, _, _, dense = piper.ik_track(targets, joints[0], restarts=4)
    _, _, _, sparse = piper.ik_track(targets[::12], joints[0], restarts=4)

    assert dense.mean() <= sparse.mean()


def test_ik_track_follows_a_real_trajectory() -> None:
    """Sequential warm starts must stay on the demonstrated kinematic branch.

    A single-shot IK from a stale seed drifts by radians on this data; this is
    the guard that experiment S5's joint targets stay continuous. Joint
    agreement is checked away from wrist singularities, where a matched pose
    does not pin the joints (see
    :func:`test_min_singular_value_collapses_at_wrist_singularity`).
    """
    states, _ = _load_dataset_frames(num_episodes=1)
    joints = states[:600:4, 0:6]
    targets = piper.fk(joints)

    solved, pos_err, rot_err, jumps = piper.ik_track(targets, joints[0])

    assert pos_err.max() < 1e-4
    assert rot_err.max() < 1e-3
    assert not jumps.any()

    well_conditioned = piper.min_singular_value(joints) > 1e-2
    assert well_conditioned.mean() > 0.5, "trajectory is degenerate"
    assert np.abs(solved - joints)[well_conditioned].max() < 1e-2
