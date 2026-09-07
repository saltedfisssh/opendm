"""Bimanual AgileX Piper cloth-folding dataset registrations.

These are the rungs of a representation ablation: the same 1255 teleoperation
episodes converted into progressively more UMI-like state and action spaces, so
that each rung differs from the previous one in exactly one variable. See
``script/piper_lerobot_to_jsonl.py`` for how the state vectors are produced.

======  ================================  ==============================================
Rung    State                             Variable changed vs the previous rung
======  ================================  ==============================================
``s0``  joint angles, 14D                 baseline
``s1``  per-arm-base EEF, 14D             joint space -> end-effector space
``s2``  per-arm-base EEF, 14D             action delta convention -> UMI body frame
``s3``  unified-frame EEF + pairing, 23D  one shared frame + explicit arm coupling
``s5``  joints from estimated base, 14D   known robot base -> base estimated from data
======  ================================  ==============================================

``s1`` and ``s2`` deliberately share the same on-disk data: they differ only in
``--data-config.relative-mode``, which is what isolates the delta convention.
``s3a`` is an optional control that unifies the frame *without* adding the
inter-gripper pairing, to separate those two effects if ``s3`` moves the needle.
``s4`` is not a dataset: world-frame invariance is asserted in
``tests/test_action_frames.py`` instead, because a frame-invariant pipeline
produces bit-identical tensors under any world frame.

Every rung must keep a distinct dataset name. ``DM05DataConfig.norm_stats_path``
hashes only the dataset name and the action transform, not ``state_desc``, so
reusing a name across two state conventions would silently share one set of
normalization statistics.
"""

from opendm.constants.robot import ROBOT_STATE_DESCS, RobotStateDesc, RobotType
from opendm.dataset.register import register_dataset

DATA_ROOT = "./data/piper_fold_cloth"
CAPTURE_ROOT = "/mnt/xiaoyu_teleop_data/piper/20260907/piper_fold_cloth_in_place"

IMAGE_KEYS = ["images_1", "images_2", "images_3"]
IMAGE_PROMPTS = ["Head", "Left wrist", "Right wrist"]

PIPER_JOINT_STATE_DESC = ROBOT_STATE_DESCS[RobotType.PIPER_DUAL]

# Per arm: xyz + axis-angle + gripper width.
PIPER_EEF_STATE_DESC = (
    [RobotStateDesc.EEF] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.EEF] * 6
    + [RobotStateDesc.GRIPPER]
)

# The trailing block is ``T_left^-1 @ T_right`` as xyz + 6D rotation. It is an
# observation-only feature, so it is tagged AUX and excluded from action targets.
PIPER_EEF_PAIR_STATE_DESC = PIPER_EEF_STATE_DESC + [RobotStateDesc.AUX] * 9


def _entry(representation: str, state_desc: list, control_mode: str) -> dict:
    return {
        # Only the train split is registered. Held-out episodes live in a sibling
        # directory because ``JsonlDataset`` globs recursively and would otherwise
        # pull them into training.
        "jsonl_dir": f"{DATA_ROOT}/{representation}/jsonl/train",
        # Frames are referenced in place against the read-only capture mount, so
        # no video is duplicated per representation.
        "image_dir": CAPTURE_ROOT,
        "image_keys": IMAGE_KEYS,
        "image_prompts": IMAGE_PROMPTS,
        "robot_type": RobotType.PIPER_DUAL,
        "state_desc": state_desc,
        "control_mode": control_mode,
    }


def held_out_dir(representation: str) -> str:
    """Return the held-out JSONL directory for a representation, for evaluation."""
    return f"{DATA_ROOT}/{representation}/jsonl/held_out"


register_dataset(
    {
        # S0: joint-space baseline.
        "s0": _entry("joint", PIPER_JOINT_STATE_DESC, "joint"),
        # S1/S2: end-effector poses in each arm's own base frame. Same bytes on
        # disk; the rungs differ only in how action deltas are formed.
        "s1": _entry("eef_local", PIPER_EEF_STATE_DESC, "eef"),
        "s2": _entry("eef_local", PIPER_EEF_STATE_DESC, "eef"),
        # S3: one shared frame for both arms, plus the inter-gripper pose.
        "s3": _entry("eef_unified_pair", PIPER_EEF_PAIR_STATE_DESC, "eef"),
        # S3a: unified frame only, to isolate it from the pairing feature.
        "s3a": _entry("eef_unified", PIPER_EEF_STATE_DESC, "eef"),
        # S5: joints recovered by IK from a base pose estimated from the data.
        "s5": _entry("joint_estimated_base", PIPER_JOINT_STATE_DESC, "joint"),
    },
    prefix="piper_fold",
)
