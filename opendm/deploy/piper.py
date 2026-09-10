"""Piper wire contract and SI-unit representation adapters (NumPy only)."""

import numpy as np
from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
from opendm.data import se3
from opendm.kinematics import piper

RUNGS = ("s0", "s1", "s2", "s3", "s3a", "s5")
PROTOCOL = "piper-relative-v1"


def contract(rung):
    if rung not in RUNGS:
        raise ValueError(f"Unknown Piper rung: {rung}")
    return dict(
        protocol=PROTOCOL,
        rung=rung,
        state_dim=23 if rung == "s3" else 14,
        action_dim=20 if rung in ("s2", "s3", "s3a") else 14,
        action_encoding="denormalized_relative",
        fps=30,
    )


def finite_array(value, shape):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Expected finite array of shape {shape}, got {value.shape}")
    return value


class Representation:
    def __init__(self, rung, bases=None):
        self.spec = contract(rung)
        self.rung = rung
        self.previous = None
        self.virtual_joints = None
        self.real_bases = np.stack([np.eye(4), piper.T_RIGHT_BASE_TO_LEFT_BASE])
        self.bases = None
        if rung == "s5":
            if bases is None:
                raise ValueError(
                    "S5 requires the estimated_bases.json used for training"
                )
            self.bases = np.stack(
                [
                    se3.make_transform(
                        finite_array(bases[a]["position"], (3,)),
                        se3.rotvec_to_mat(
                            finite_array([0, 0, bases[a]["yaw_rad"]], (3,))
                        ),
                    )
                    for a in ("left", "right")
                ]
            )

    def observe(self, joints):
        joints = finite_array(joints, (14,)).copy()
        if self.rung == "s0":
            return joints
        poses = piper.fk(joints.reshape(2, 7)[:, :6])
        if self.rung == "s5":
            for a in range(2):
                target = (
                    se3.transform_inverse(self.bases[a]) @ self.real_bases[a] @ poses[a]
                )
                seed = (
                    joints[a * 7 : a * 7 + 6]
                    if self.virtual_joints is None
                    else self.virtual_joints[a]
                )
                q, pe, re = piper.ik_multistart(target, [seed, piper.neutral_joints()])
                if pe > 0.003 or re > 0.03:
                    raise ValueError("S5 observation unreachable in virtual base")
                joints[a * 7 : a * 7 + 6] = q
            self.virtual_joints = joints.reshape(2, 7)[:, :6].copy()
            return joints
        if self.rung in ("s3", "s3a"):
            poses = self.real_bases @ poses
        state = np.concatenate(
            [
                np.r_[se3.transform_to_pos_rotvec(poses[a]), joints[a * 7 + 6]]
                for a in range(2)
            ]
        )
        if self.previous is not None:
            for start in (3, 10):
                state[start : start + 3] = se3.unwrap_rotvec_sequence(
                    np.stack(
                        [self.previous[start : start + 3], state[start : start + 3]]
                    )
                )[-1]
        self.previous = state.copy()
        if self.rung == "s3":
            state = np.r_[
                state,
                se3.transform_to_pos_rot6d(se3.transform_inverse(poses[0]) @ poses[1]),
            ]
        return state

    def decode(self, state, actions):
        state = finite_array(state, (self.spec["state_dim"],))
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or not len(actions):
            raise ValueError("Expected nonempty action chunk")
        actions = finite_array(actions, (len(actions), self.spec["action_dim"]))
        output = np.empty((len(actions), 14))
        for a in range(2):
            s = slice(a * 7, a * 7 + 6)
            if self.rung in ("s0", "s5"):
                q = state[s] + actions[:, s]
                output[:, a * 7 + 6] = actions[:, a * 7 + 6]
                if self.rung == "s0":
                    output[:, s] = q
                    continue
                pose = (
                    se3.transform_inverse(self.real_bases[a])
                    @ self.bases[a]
                    @ piper.fk(q)
                )
            else:
                anchor = se3.pos_rotvec_to_transform(state[s])
                if self.spec["action_dim"] == 20:
                    block = actions[:, a * 10 : a * 10 + 9]
                    # Reject undefined 6D rotations before Gram-Schmidt can hide them.
                    if (np.linalg.norm(block[:, 3:6], axis=-1) < 1e-6).any() or (
                        np.linalg.norm(np.cross(block[:, 3:6], block[:, 6:9]), axis=-1)
                        < 1e-6
                    ).any():
                        raise ValueError("Degenerate 6D rotation")
                    pose = anchor @ se3.pos_rot6d_to_transform(block)
                    output[:, a * 7 + 6] = actions[:, a * 10 + 9]
                else:
                    pose = se3.make_transform(
                        state[s][:3] + actions[:, s][:, :3],
                        se3.rotvec_to_mat(actions[:, s][:, 3:]) @ anchor[:3, :3],
                    )
                    output[:, a * 7 + 6] = actions[:, a * 7 + 6]
                if self.rung in ("s3", "s3a"):
                    pose = se3.transform_inverse(self.real_bases[a]) @ pose
            output[:, s] = np.concatenate(
                [pose[:, :3, 3], se3.mat_to_rpy(pose[:, :3, :3])], axis=-1
            )
        return finite_array(output, (len(actions), 14))


def validate_motion(
    commands,
    current,
    joint_mode,
    max_joint=0.15,
    max_position=0.025,
    max_rotation=0.15,
    max_gripper=0.02,
):
    """Validate the entire executed prefix before sending either arm a command."""
    commands = np.asarray(commands)
    finite_array(commands, (len(commands), 14))
    current = finite_array(current, (14,))
    if ((commands[:, [6, 13]] < 0) | (commands[:, [6, 13]] > 0.08)).any():
        raise ValueError("Gripper width outside [0, 0.08] m")
    trajectory = np.vstack([current, commands]).reshape(-1, 2, 7)
    if (np.abs(np.diff(trajectory[:, :, 6], axis=0)) > max_gripper).any():
        raise ValueError("Gripper step exceeds limit")
    if joint_mode:
        limits = np.asarray(
            [ROBOT_JOINT_LIMIT_PRESET_RAD["piper"][f"joint{i}"] for i in range(1, 7)]
        )
        targets = trajectory[1:, :, :6]
        if ((targets < limits[:, 0]) | (targets > limits[:, 1])).any():
            raise ValueError("Joint target outside SDK Piper limits")
        if (np.abs(np.diff(trajectory[:, :, :6], axis=0)) > max_joint).any():
            raise ValueError("Joint step exceeds limit")
    else:
        if (
            np.linalg.norm(np.diff(trajectory[:, :, :3], axis=0), axis=-1)
            > max_position
        ).any():
            raise ValueError("Cartesian step exceeds limit")
        rot = se3.rpy_to_mat(trajectory[:, :, 3:6])
        angles = np.linalg.norm(
            se3.mat_to_rotvec(np.swapaxes(rot[:-1], -1, -2) @ rot[1:]), axis=-1
        )
        if (angles > max_rotation).any():
            raise ValueError("Rotation step exceeds limit")
        # SDK move_p accepts canonical Euler angles only. Reject branch crossings
        # instead of passing unwrapped angles outside its documented domain.
        if (np.abs(np.diff(trajectory[:, :, 3:6], axis=0)) > np.pi).any():
            raise ValueError("Euler branch crossing: reposition before rollout")
