"""Regression tests for the SE(3) helpers in ``opendm.data.se3``.

The reference for every rotation conversion is ``scipy.spatial.transform``,
which is already a project dependency.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from opendm.data import se3

NUM_SAMPLES = 512


@pytest.fixture
def rotations() -> Rotation:
    return Rotation.random(NUM_SAMPLES, random_state=17)


@pytest.fixture
def transforms(rotations: Rotation) -> np.ndarray:
    rng = np.random.default_rng(17)
    return se3.make_transform(
        rng.normal(size=(NUM_SAMPLES, 3)),
        rotations.as_matrix(),
    )


def test_quat_to_mat_matches_scipy(rotations: Rotation) -> None:
    got = se3.quat_to_mat(rotations.as_quat())
    assert np.allclose(got, rotations.as_matrix(), atol=1e-12)


def test_rotvec_to_mat_matches_scipy(rotations: Rotation) -> None:
    got = se3.rotvec_to_mat(rotations.as_rotvec())
    assert np.allclose(got, rotations.as_matrix(), atol=1e-12)


def test_rpy_matches_scipy_extrinsic_xyz() -> None:
    """The repo convention is ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

    That is scipy's extrinsic ``"xyz"``, and it is the convention the AgileX SDK
    reports flange orientation in, so a mismatch here silently corrupts every
    EEF representation.
    """
    rng = np.random.default_rng(3)
    rpy = np.stack(
        [
            rng.uniform(-np.pi, np.pi, NUM_SAMPLES),
            rng.uniform(-1.5, 1.5, NUM_SAMPLES),
            rng.uniform(-np.pi, np.pi, NUM_SAMPLES),
        ],
        axis=-1,
    )
    expected = Rotation.from_euler("xyz", rpy).as_matrix()
    assert np.allclose(se3.rpy_to_mat(rpy), expected, atol=1e-12)
    assert np.allclose(se3.rpy_to_mat(se3.mat_to_rpy(expected)), expected, atol=1e-10)


@pytest.mark.parametrize(
    ("encode", "decode"),
    [
        (se3.mat_to_quat, se3.quat_to_mat),
        (se3.mat_to_rotvec, se3.rotvec_to_mat),
        (se3.mat_to_rot6d, se3.rot6d_to_mat),
    ],
)
def test_rotation_roundtrips(rotations: Rotation, encode, decode) -> None:
    mat = rotations.as_matrix()
    assert np.allclose(decode(encode(mat)), mat, atol=1e-12)


@pytest.mark.parametrize(
    ("encode", "decode"),
    [
        (se3.transform_to_pos_quat, se3.pos_quat_to_transform),
        (se3.transform_to_pos_rotvec, se3.pos_rotvec_to_transform),
        (se3.transform_to_pos_rot6d, se3.pos_rot6d_to_transform),
    ],
)
def test_transform_layout_roundtrips(transforms: np.ndarray, encode, decode) -> None:
    assert np.allclose(decode(encode(transforms)), transforms, atol=1e-12)


def test_mat_to_quat_stable_at_half_turns() -> None:
    """Half turns are where a naive trace-based conversion loses precision."""
    axes = np.concatenate([np.eye(3), -np.eye(3), np.ones((1, 3)) / np.sqrt(3.0)])
    mat = Rotation.from_rotvec(axes * np.pi).as_matrix()
    assert np.allclose(se3.quat_to_mat(se3.mat_to_quat(mat)), mat, atol=1e-12)


def test_rotvec_to_mat_identity_at_zero() -> None:
    assert np.allclose(se3.rotvec_to_mat(np.zeros((4, 3))), np.eye(3), atol=0.0)


def test_canonical_quat_pins_sign(rotations: Rotation) -> None:
    quat = rotations.as_quat()
    canonical = se3.canonical_quat(quat)
    assert (canonical[:, 3] >= 0.0).all()
    # Same rotation, just the other cover of SU(2).
    assert np.allclose(se3.quat_to_mat(canonical), se3.quat_to_mat(quat), atol=1e-12)


def test_transform_inverse(transforms: np.ndarray) -> None:
    identity = transforms @ se3.transform_inverse(transforms)
    assert np.allclose(identity, np.eye(4), atol=1e-12)


def test_relative_and_absolute_are_inverses(transforms: np.ndarray) -> None:
    anchor = np.roll(transforms, 1, axis=0)
    relative = se3.relative_transform(anchor, transforms)
    assert np.allclose(se3.absolute_transform(anchor, relative), transforms, atol=1e-12)


def test_relative_transform_is_world_frame_invariant(transforms: np.ndarray) -> None:
    """The property the whole UMI action representation rests on.

    ``anchor^-1 @ target`` is unchanged by any rigid re-expression of the world
    frame, which is why an arbitrary SLAM origin and an uncalibrated robot base
    both drop out of the action targets.
    """
    anchor = np.roll(transforms, 1, axis=0)
    baseline = se3.relative_transform(anchor, transforms)

    world = se3.make_transform(
        np.array([1.3, -0.7, 4.2]),
        Rotation.from_rotvec([0.4, -1.1, 2.3]).as_matrix(),
    )
    shifted = se3.relative_transform(world @ anchor, world @ transforms)
    assert np.allclose(shifted, baseline, atol=1e-12)


def test_transform_points_matches_matrix_product(transforms: np.ndarray) -> None:
    rng = np.random.default_rng(5)
    points = rng.normal(size=(NUM_SAMPLES, 3))
    homogeneous = np.concatenate([points, np.ones((NUM_SAMPLES, 1))], axis=-1)
    expected = np.einsum("nij,nj->ni", transforms, homogeneous)[:, :3]
    assert np.allclose(se3.transform_points(transforms, points), expected, atol=1e-12)


def _sweep_through_pi(num: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """A trajectory whose rotation angle crosses ``pi``, the failure case."""
    axis = np.array([0.3, -0.5, 0.81])
    axis /= np.linalg.norm(axis)
    angles = np.linspace(np.pi - 0.25, np.pi + 0.25, num)
    mat = Rotation.from_rotvec(angles[:, None] * axis).as_matrix()
    return mat, se3.mat_to_rotvec(mat)


def test_unwrap_rotvec_removes_the_pi_discontinuity() -> None:
    mat, rotvec = _sweep_through_pi()
    # Without unwrapping the encoding jumps by nearly 2*pi.
    assert np.abs(np.diff(rotvec, axis=0)).max() > 4.0

    unwrapped = se3.unwrap_rotvec_sequence(rotvec)
    assert np.abs(np.diff(unwrapped, axis=0)).max() < 0.05
    # The chart changed; the rotation did not.
    assert np.allclose(se3.rotvec_to_mat(unwrapped), mat, atol=1e-12)


def test_unwrap_rotvec_is_noop_on_continuous_input() -> None:
    axis = np.array([0.3, -0.5, 0.81])
    axis /= np.linalg.norm(axis)
    rotvec = se3.mat_to_rotvec(
        Rotation.from_rotvec(np.linspace(0.0, 1.0, 32)[:, None] * axis).as_matrix()
    )
    assert np.array_equal(se3.unwrap_rotvec_sequence(rotvec), rotvec)


def test_unwrap_rotvec_handles_batch_axes() -> None:
    """Each arm in a bimanual sequence must unwrap independently."""
    mat, rotvec = _sweep_through_pi()
    stacked = np.stack([rotvec, rotvec[::-1]], axis=1)

    unwrapped = se3.unwrap_rotvec_sequence(stacked)

    assert unwrapped.shape == stacked.shape
    assert np.abs(np.diff(unwrapped, axis=0)).max() < 0.05
    assert np.allclose(
        se3.rotvec_to_mat(unwrapped), se3.rotvec_to_mat(stacked), atol=1e-12
    )
