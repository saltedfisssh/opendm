"""Train-only RoboTwin representation registrations; videos remain in place."""

from opendm.constants.robot import ROBOT_STATE_DESCS, RobotStateDesc, RobotType
from opendm.dataset.register import register_dataset

SOURCE = "/kpfs_ssd/data/wzy/data/Dexmal/robotwin2-full/robotwin2.0"
DATA_ROOT = "/kpfs_ssd/data/wzy/data/Dexmal/robotwin2-full/robotwin2_ablation"
REPRESENTATIONS = {
    "s0": "joint",
    "s1": "eef_local",
    "s2": "eef_local",
    "s2_pair": "eef_local_pair",
    "s3a": "eef_unified",
    "s3": "eef_unified_pair",
    "s4": "eef_gravity_random",
}
RELATIVE_MODES = {r: "vector" if r in ("s0", "s1") else "se3" for r in REPRESENTATIONS}
EEF_STATE_DESC = ([RobotStateDesc.EEF] * 6 + [RobotStateDesc.GRIPPER]) * 2
register_dataset(
    {
        r: {
            "jsonl_dir": f"{DATA_ROOT}/{rep}/jsonl/train",
            "image_dir": f"{SOURCE}/video",
            "image_keys": ["images_1", "images_2", "images_3"],
            "image_prompts": ["Head", "Left wrist", "Right wrist"],
            "robot_type": RobotType.ALOHA_ROBOTWIN2,
            "state_desc": (
                ROBOT_STATE_DESCS[RobotType.ALOHA_ROBOTWIN2]
                if r == "s0"
                else EEF_STATE_DESC
                + ([RobotStateDesc.AUX] * 9 if r in ("s3", "s2_pair") else [])
            ),
            "control_mode": "joint" if r == "s0" else "eef",
        }
        for r, rep in REPRESENTATIONS.items()
    },
    prefix="robotwin2_ablation",
)
