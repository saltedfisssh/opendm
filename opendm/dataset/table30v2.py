"""Register converted RoboChallenge Table30 v2 datasets.

Run ``script/convert_table30v2.py`` first.  The converter writes a manifest
containing one dataset per task and one aggregate dataset per embodiment.
"""

import json
import os
from pathlib import Path

from opendm.constants.robot import RobotStateDesc
from opendm.dataset.register import register_dataset


DEFAULT_MANIFEST = Path("./data/table30v2_dexdata_binary/manifest.json")
MANIFEST_PATH = Path(os.getenv("OPENDM_TABLE30V2_MANIFEST", DEFAULT_MANIFEST))


def _state_desc(state_dim: int) -> list[RobotStateDesc]:
    if state_dim == 7:
        return [RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER]
    if state_dim == 14:
        arm = [RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER]
        return arm + arm
    raise ValueError(f"unsupported Table30v2 state dimension: {state_dim}")


def _load_datasets(manifest_path: Path) -> dict:
    if not manifest_path.is_file():
        return {}
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format_version") != 1:
        raise ValueError(f"unsupported Table30v2 manifest: {manifest_path}")

    registered = {}
    for name, config in manifest.get("datasets", {}).items():
        registered[name] = {
            "jsonl_dir": config["jsonl_dir"],
            "image_dir": config["image_dir"],
            "image_keys": config["image_keys"],
            "state_desc": _state_desc(config["state_dim"]),
            "embodiment": config["embodiment"],
            "tasks": config["tasks"],
        }
    return registered


register_dataset(_load_datasets(MANIFEST_PATH))
