#!/usr/bin/env python3
"""Run in RoboTwin's .venv: independently compare NumPy URDF FK to SAPIEN.

Visual/collision elements are removed in a temporary URDF; kinematic joints,
origins, limits and inertias are unchanged. No rendering or asset writes occur.
"""

import argparse
import json
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import sapien.core as sapien

from opendm.kinematics.robotwin2 import RoboTwinKinematics, ROBOTWIN_ROOT
from opendm.data import se3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", default=str(ROBOTWIN_ROOT / "assets"))
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--out")
    args = p.parse_args()
    if args.samples < 1:
        p.error("--samples must be positive")
    kin = RoboTwinKinematics(args.assets)
    tree = ET.parse(kin.urdf_path)
    for link in tree.getroot().findall("link"):
        for node in list(link):
            if node.tag in ("visual", "collision"):
                link.remove(node)
    engine = sapien.Engine()
    scene = engine.create_scene()
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    with tempfile.TemporaryDirectory() as directory:
        path = directory + "/kinematics.urdf"
        tree.write(path)
        robot = loader.load(path)
    active = {joint.name: i for i, joint in enumerate(robot.get_active_joints())}
    links = {link.name: link for link in robot.get_links()}
    rng = np.random.default_rng(0)
    errors = []
    for sample in range(args.samples):
        q = np.zeros(robot.dof)
        for names in kin.names:
            for name in names:
                limit = kin.joints[name].find("limit")
                q[active[name]] = (
                    0
                    if sample == 0
                    else rng.uniform(
                        float(limit.get("lower")), float(limit.get("upper"))
                    )
                )
        robot.set_qpos(q)
        for arm in range(2):
            target = links[kin.tips[arm]].get_pose().to_transformation_matrix()
            base = links[kin.bases[arm]].get_pose().to_transformation_matrix()
            local = se3.transform_inverse(base) @ target
            calculated = kin.fk(q[[active[n] for n in kin.names[arm]]], arm)
            errors.append(float(np.max(np.abs(local - calculated))))
    report = {
        "samples": args.samples,
        "arms": 2,
        "max_matrix_abs_error": max(errors),
        "tolerance": 2e-6,
        "eef": kin.tips,
        "scope": "installed URDF vs SAPIEN; does not establish export-version provenance",
    }
    print(json.dumps(report, indent=2))
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    if max(errors) >= report["tolerance"]:
        raise SystemExit("FK validation failed")


if __name__ == "__main__":
    main()
