from enum import Enum


class ActionMode(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class RobotStateDesc(Enum):
    JOINT = "joint"
    EEF = "eef"
    GRIPPER = "gripper"


class RobotType(Enum):
    DOS_W1 = "DOS W1"
    ARX5 = "ARX5"
    UR5 = "UR5"
    ALOHA = "ALOHA"
    FRANKA = "Franka"
    PIPER = "Piper"
    ALOHA_ROBOTWIN2 = "Aloha RoboTwin2"


ROBOT_STATE_DESCS = {
    RobotType.DOS_W1: [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER],
    RobotType.ALOHA_ROBOTWIN2: [RobotStateDesc.JOINT] * 6,
    RobotType.ARX5: [RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER],
    RobotType.UR5: [RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER],
    RobotType.ALOHA: [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER],
    RobotType.PIPER: [RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER],
}
