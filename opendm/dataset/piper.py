"""Piper datasets converted from LeRobot v2.1."""

from opendm.constants.robot import RobotStateDesc
from opendm.dataset.register import register_dataset

PIPER_STATE_DESC = [RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER]

register_dataset(
    {
        "arrange_the_fruits": {
            "jsonl_dir": "./data/arrange_the_fruits_piper/opendm",
            # Converted JSONL video URLs are relative to the LeRobot dataset root.
            "image_dir": "./data/arrange_the_fruits_piper/merged_binarized",
            "image_keys": ["images_1", "images_2"],
            "state_desc": PIPER_STATE_DESC,
        },
        "arrange_the_fruits_no_binarized": {
            "jsonl_dir": "./data/arrange_the_fruits_piper/opendm_no_binarized",
            # Converted JSONL video URLs are relative to the LeRobot dataset root.
            "image_dir": "./data/arrange_the_fruits_piper/merged",
            "image_keys": ["images_1", "images_2"],
            "state_desc": PIPER_STATE_DESC, 
        }
    },
    prefix="piper",
)
