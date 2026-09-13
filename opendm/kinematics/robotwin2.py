"""URDF FK for the installed RoboTwin aloha-agilex embodiment.

EEF means the raw move_group link frame (fl_link6/fr_link6), NOT the
simulator's transformed gripper-centre control TCP. No meshes are loaded.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from opendm.data import se3

ROBOTWIN_ROOT = Path("/kpfs_ssd/data/wzy/dexbotic-benchmark/RoboTwin")


class RoboTwinKinematics:
    def __init__(self, assets=ROBOTWIN_ROOT / "assets"):
        self.config_path = Path(assets) / "embodiments/aloha-agilex/config.yml"
        self.config = yaml.safe_load(self.config_path.read_text())
        if not self.config.get("dual_arm"):
            raise ValueError("Only the single-articulation aloha-agilex is supported")
        self.urdf_path = self.config_path.parent / self.config["urdf_path"]
        root = ET.parse(self.urdf_path).getroot()
        self.joints = {j.attrib["name"]: j for j in root.findall("joint")}
        self.parents = {j.find("child").attrib["link"]: j for j in self.joints.values()}
        self.names = self.config["arm_joints_name"]
        self.tips = self.config["move_group"]
        self.bases = [
            self.joints[n[0]].find("parent").attrib["link"] for n in self.names
        ]
        self.root_to_bases = [self._fk(b, {}, ()) for b in self.bases]

    def _fk(self, link, values, shape):
        chain = []
        while link in self.parents:
            joint = self.parents[link]
            chain.append(joint)
            link = joint.find("parent").attrib["link"]
        result = np.broadcast_to(np.eye(4), (*shape, 4, 4)).copy()
        for joint in reversed(chain):
            origin = joint.find("origin")
            xyz = np.fromstring(
                origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" "
            )
            rpy = np.fromstring(
                origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" "
            )
            result = result @ se3.make_transform(
                xyz, Rotation.from_euler("xyz", rpy).as_matrix()
            )
            kind = joint.attrib["type"]
            if kind == "fixed":
                continue
            name = joint.attrib["name"]
            if name not in values:
                raise ValueError(f"Unspecified moving ancestor: {name}")
            axis_node = joint.find("axis")
            axis = np.fromstring(
                axis_node.get("xyz") if axis_node is not None else "1 0 0", sep=" "
            )
            axis = axis / np.linalg.norm(axis)
            q = np.asarray(values[name])
            motion = np.broadcast_to(np.eye(4), (*shape, 4, 4)).copy()
            if kind in ("revolute", "continuous"):
                motion[..., :3, :3] = se3.rotvec_to_mat(q[..., None] * axis)
            elif kind == "prismatic":
                motion[..., :3, 3] = q[..., None] * axis
            else:
                raise ValueError(f"Unsupported URDF joint type: {kind}")
            result = result @ motion
        return result

    def fk(self, joints, arm, unified=False):
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape[-1] != 6 or not np.isfinite(joints).all():
            raise ValueError("Expected finite (..., 6) joint positions")
        pose = self._fk(
            self.tips[arm],
            dict(zip(self.names[arm], np.moveaxis(joints, -1, 0))),
            joints.shape[:-1],
        )
        # W is the left arm base, with its original +z sign. Shared root pose
        # cancels exactly, even if the entire fixed dual-arm rig is relocated.
        base = self.root_to_bases[0 if unified else arm]
        return se3.transform_inverse(base) @ pose
