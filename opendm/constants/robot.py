from enum import Enum

# Soft tokens per history image (``<unused0>`` placeholders / pooled vision tokens).
HISTORY_TOKENS_PER_IMAGE = 16
# 2D adaptive-pool spatial size; ``HISTORY_POOL_SIZE ** 2 == HISTORY_TOKENS_PER_IMAGE``.
HISTORY_POOL_SIZE = int(HISTORY_TOKENS_PER_IMAGE**0.5)
assert HISTORY_POOL_SIZE * HISTORY_POOL_SIZE == HISTORY_TOKENS_PER_IMAGE


class ActionMode(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class RobotStateDesc(Enum):
    JOINT = "joint"
    EEF = "eef"
    GRIPPER = "gripper"
    # Observation-only context dimensions. They are never action targets and are
    # never delta-encoded; action transforms skip them. Used for derived features
    # such as the inter-gripper relative pose in bimanual end-effector setups.
    AUX = "aux"


class RobotType(Enum):
    DOS_W1 = "DOS W1"
    FRANKA = "Franka"
    ALOHA = "Aloha"
    ALOHA_ROBOTWIN2 = "Aloha RoboTwin2"
    SO101 = "SO101"
    ARX5 = "ARX5"
    UR5 = "UR5"
    PIPER_DUAL = "Piper Dual"


ROBOT_STATE_DESCS = {
    RobotType.DOS_W1: [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER],
    RobotType.ALOHA: [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER],
    RobotType.ALOHA_ROBOTWIN2: [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER],
    RobotType.SO101: [RobotStateDesc.JOINT] * 5 + [RobotStateDesc.GRIPPER],
    RobotType.ARX5: [RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER],
    RobotType.UR5: [RobotStateDesc.EEF] * 6 + [RobotStateDesc.GRIPPER],
    # Bimanual AgileX Piper. The joint layout is the default; the end-effector
    # variants of this embodiment declare their own descriptors at registration
    # time (see ``opendm/dataset/piper_dual.py``).
    RobotType.PIPER_DUAL: [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER],
}
