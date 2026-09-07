"""SE(3) and rotation utilities for end-effector action representations.

This module holds the frame math needed by end-effector (EEF) action
representations: homogeneous transforms, the continuous 6D rotation
representation of Zhou et al., and conversions between the layouts used on
disk (``xyz + quaternion``), in training samples (``xyz + axis-angle``) and in
UMI-style relative actions (``translation + 6D rotation``).

All functions are numpy-only, operate on trailing dimensions, and accept
arbitrary leading batch shapes.

Conventions
-----------
Quaternions are ``(x, y, z, w)``, matching both the AgileX SDK
(``pyAgxArm.utiles.tf.euler_convert_quat``) and ``scipy``. Axis-angle vectors
are rotation vectors (``rotvec``) whose norm is the rotation angle. Roll /
pitch / yaw compose as ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``, consistent with
``opendm.data.transforms.rpy_to_axis_angle``.
"""

import numpy as np

_EPS = 1e-8


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    """Scale ``(..., 4)`` xyzw quaternions to unit norm."""
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, _EPS)


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """Convert ``(..., 4)`` xyzw quaternions to ``(..., 3, 3)`` rotations."""
    x, y, z, w = np.moveaxis(normalize_quat(quat), -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    mat = np.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        axis=-1,
    )
    return mat.reshape(*mat.shape[:-1], 3, 3)


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert ``(..., 3, 3)`` rotations to ``(..., 4)`` xyzw quaternions.

    Uses the branch-free formulation of Shepperd's method: build all four
    candidate quaternions and pick the one whose pivot is largest, which stays
    numerically stable near 180-degree rotations.
    """
    mat = np.asarray(mat, dtype=np.float64)
    m00, m01, m02 = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    m10, m11, m12 = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    m20, m21, m22 = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]

    # Candidate quaternions, each scaled so that one component is dominant.
    cands = np.stack(
        [
            np.stack([m21 - m12, m02 - m20, m10 - m01, 1.0 + m00 + m11 + m22], -1),
            np.stack([1.0 + m00 - m11 - m22, m01 + m10, m02 + m20, m21 - m12], -1),
            np.stack([m01 + m10, 1.0 - m00 + m11 - m22, m12 + m21, m02 - m20], -1),
            np.stack([m02 + m20, m12 + m21, 1.0 - m00 - m11 + m22, m10 - m01], -1),
        ],
        axis=-2,
    )
    pivots = np.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        axis=-1,
    )
    best = np.argmax(pivots, axis=-1)
    quat = np.take_along_axis(cands, best[..., None, None], axis=-2)[..., 0, :]
    return canonical_quat(normalize_quat(quat))


def canonical_quat(quat: np.ndarray) -> np.ndarray:
    """Flip ``(..., 4)`` xyzw quaternions to the ``w >= 0`` hemisphere.

    ``q`` and ``-q`` are the same rotation; pinning the sign keeps regression
    targets continuous instead of letting them jump between the two covers.
    """
    quat = np.asarray(quat, dtype=np.float64)
    return np.where(quat[..., 3:4] < 0.0, -quat, quat)


def rotvec_to_mat(rotvec: np.ndarray) -> np.ndarray:
    """Convert ``(..., 3)`` rotation vectors to ``(..., 3, 3)`` rotations."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    axis = rotvec / np.maximum(angle, _EPS)
    x, y, z = np.moveaxis(axis, -1, 0)
    cos = np.cos(angle)[..., 0]
    sin = np.sin(angle)[..., 0]
    one = 1.0 - cos
    mat = np.stack(
        [
            cos + x * x * one,
            x * y * one - z * sin,
            x * z * one + y * sin,
            y * x * one + z * sin,
            cos + y * y * one,
            y * z * one - x * sin,
            z * x * one - y * sin,
            z * y * one + x * sin,
            cos + z * z * one,
        ],
        axis=-1,
    ).reshape(*rotvec.shape[:-1], 3, 3)
    # Below the small-angle cutoff ``axis`` is meaningless; fall back to identity.
    tiny = (angle[..., 0] <= _EPS)[..., None, None]
    return np.where(tiny, np.eye(3), mat)


def mat_to_rotvec(mat: np.ndarray) -> np.ndarray:
    """Convert ``(..., 3, 3)`` rotations to ``(..., 3)`` rotation vectors."""
    quat = canonical_quat(mat_to_quat(mat))
    vec = quat[..., :3]
    vec_norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vec_norm, quat[..., 3:4])
    return np.where(
        vec_norm > _EPS,
        vec * (angle / np.maximum(vec_norm, _EPS)),
        np.zeros_like(vec),
    )


def rpy_to_mat(rpy: np.ndarray) -> np.ndarray:
    """Convert ``(..., 3)`` roll/pitch/yaw to ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    rpy = np.asarray(rpy, dtype=np.float64)
    roll, pitch, yaw = np.moveaxis(rpy, -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    mat = np.stack(
        [
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        ],
        axis=-1,
    )
    return mat.reshape(*rpy.shape[:-1], 3, 3)


def mat_to_rpy(mat: np.ndarray) -> np.ndarray:
    """Invert :func:`rpy_to_mat`, returning ``(..., 3)`` roll/pitch/yaw."""
    mat = np.asarray(mat, dtype=np.float64)
    pitch = -np.arcsin(np.clip(mat[..., 2, 0], -1.0, 1.0))
    roll = np.arctan2(mat[..., 2, 1], mat[..., 2, 2])
    yaw = np.arctan2(mat[..., 1, 0], mat[..., 0, 0])
    return np.stack([roll, pitch, yaw], axis=-1)


def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """Take the first two rows of ``(..., 3, 3)`` rotations as a 6D vector.

    This is the continuous rotation representation of Zhou et al., which the
    UMI line of work uses in place of Euler angles or axis-angle to avoid
    wrap-around and gimbal-lock discontinuities in regression targets.
    """
    mat = np.asarray(mat, dtype=np.float64)
    return np.concatenate([mat[..., 0, :], mat[..., 1, :]], axis=-1)


def rot6d_to_mat(rot6d: np.ndarray) -> np.ndarray:
    """Recover ``(..., 3, 3)`` rotations from ``(..., 6)`` via Gram-Schmidt."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1, a2 = rot6d[..., :3], rot6d[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), _EPS)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / np.maximum(np.linalg.norm(a2, axis=-1, keepdims=True), _EPS)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def unwrap_rotvec_sequence(rotvec: np.ndarray, axis: int = 0) -> np.ndarray:
    """Remove ``2*pi`` sign flips along a sequence of rotation vectors.

    A rotation of angle ``theta`` about axis ``n`` is equally well written as
    ``theta * n`` or ``(theta - 2*pi) * n``, and :func:`mat_to_rotvec` always
    returns the first. Recorded Piper flange rotations sit at 2.1-2.3 rad with a
    tail reaching 3.08 rad, i.e. within 0.06 rad of ``pi``, so a trajectory that
    crosses ``pi`` produces a discontinuity of nearly ``2*pi`` in an otherwise
    smooth signal. That corrupts binned state tokens and any finite-difference
    action target computed from them.

    This picks, per frame, whichever of the two representations is closer to the
    previous frame. The decoded rotation is unchanged; only the chart is.

    Args:
        rotvec: Rotation vectors with a time axis.
        axis: The time axis to unwrap along.

    Returns:
        Array of the same shape, continuous along ``axis``.
    """
    rotvec = np.moveaxis(np.asarray(rotvec, dtype=np.float64).copy(), axis, 0)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    unit = rotvec / np.maximum(angle, _EPS)
    # The alternative representative of the same rotation.
    alternate = rotvec - 2.0 * np.pi * unit

    for t in range(1, rotvec.shape[0]):
        take_alternate = np.linalg.norm(
            alternate[t] - rotvec[t - 1], axis=-1
        ) < np.linalg.norm(rotvec[t] - rotvec[t - 1], axis=-1)
        rotvec[t] = np.where(take_alternate[..., None], alternate[t], rotvec[t])

    return np.moveaxis(rotvec, 0, axis)


def make_transform(pos: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Pack ``(..., 3)`` positions and ``(..., 3, 3)`` rotations into ``(..., 4, 4)``."""
    pos = np.asarray(pos, dtype=np.float64)
    mat = np.asarray(mat, dtype=np.float64)
    batch = np.broadcast_shapes(pos.shape[:-1], mat.shape[:-2])
    out = np.zeros((*batch, 4, 4), dtype=np.float64)
    out[..., :3, :3] = mat
    out[..., :3, 3] = pos
    out[..., 3, 3] = 1.0
    return out


def transform_inverse(transform: np.ndarray) -> np.ndarray:
    """Invert ``(..., 4, 4)`` rigid transforms without a general matrix solve."""
    transform = np.asarray(transform, dtype=np.float64)
    rot = transform[..., :3, :3]
    pos = transform[..., :3, 3]
    rot_t = np.swapaxes(rot, -1, -2)
    return make_transform(-np.einsum("...ij,...j->...i", rot_t, pos), rot_t)


def transform_compose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return ``left @ right`` for ``(..., 4, 4)`` transforms."""
    return np.asarray(left, dtype=np.float64) @ np.asarray(right, dtype=np.float64)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply ``(..., 4, 4)`` transforms to ``(..., 3)`` points."""
    transform = np.asarray(transform, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    rot = transform[..., :3, :3]
    pos = transform[..., :3, 3]
    return np.einsum("...ij,...j->...i", rot, points) + pos


def pos_quat_to_transform(pos_quat: np.ndarray) -> np.ndarray:
    """Convert ``(..., 7)`` ``[xyz, qx, qy, qz, qw]`` to ``(..., 4, 4)``."""
    pos_quat = np.asarray(pos_quat, dtype=np.float64)
    return make_transform(pos_quat[..., :3], quat_to_mat(pos_quat[..., 3:7]))


def transform_to_pos_quat(transform: np.ndarray) -> np.ndarray:
    """Convert ``(..., 4, 4)`` to ``(..., 7)`` ``[xyz, qx, qy, qz, qw]``."""
    transform = np.asarray(transform, dtype=np.float64)
    return np.concatenate(
        [transform[..., :3, 3], mat_to_quat(transform[..., :3, :3])],
        axis=-1,
    )


def pos_rotvec_to_transform(pos_rotvec: np.ndarray) -> np.ndarray:
    """Convert ``(..., 6)`` ``[xyz, rotvec]`` to ``(..., 4, 4)``."""
    pos_rotvec = np.asarray(pos_rotvec, dtype=np.float64)
    return make_transform(pos_rotvec[..., :3], rotvec_to_mat(pos_rotvec[..., 3:6]))


def transform_to_pos_rotvec(transform: np.ndarray) -> np.ndarray:
    """Convert ``(..., 4, 4)`` to ``(..., 6)`` ``[xyz, rotvec]``."""
    transform = np.asarray(transform, dtype=np.float64)
    return np.concatenate(
        [transform[..., :3, 3], mat_to_rotvec(transform[..., :3, :3])],
        axis=-1,
    )


def pos_rot6d_to_transform(pos_rot6d: np.ndarray) -> np.ndarray:
    """Convert ``(..., 9)`` ``[xyz, rot6d]`` to ``(..., 4, 4)``."""
    pos_rot6d = np.asarray(pos_rot6d, dtype=np.float64)
    return make_transform(pos_rot6d[..., :3], rot6d_to_mat(pos_rot6d[..., 3:9]))


def transform_to_pos_rot6d(transform: np.ndarray) -> np.ndarray:
    """Convert ``(..., 4, 4)`` to ``(..., 9)`` ``[xyz, rot6d]``."""
    transform = np.asarray(transform, dtype=np.float64)
    return np.concatenate(
        [transform[..., :3, 3], mat_to_rot6d(transform[..., :3, :3])],
        axis=-1,
    )


def relative_transform(anchor: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Express ``target`` in the body frame of ``anchor``: ``anchor^-1 @ target``.

    This is the UMI relative-trajectory action: it cancels both the arbitrary
    world-frame origin and any unknown robot-base calibration, because the two
    poses are measured in the same (unspecified) frame.
    """
    return transform_compose(transform_inverse(anchor), target)


def absolute_transform(anchor: np.ndarray, relative: np.ndarray) -> np.ndarray:
    """Invert :func:`relative_transform`: ``anchor @ relative``."""
    return transform_compose(anchor, relative)
