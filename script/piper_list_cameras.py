#!/usr/bin/env python3
"""List RealSense serials and color modes on the local deployment machine."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opendm.deploy.realsense import list_devices


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        devices = list_devices()
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(devices, indent=2, ensure_ascii=False))
    if not devices:
        print(
            "No RealSense devices found. Run this script on the robot host with cameras connected.",
            file=sys.stderr,
        )
        return 1
    print(
        "Pass serials in Head / Left wrist / Right wrist order to --cameras; enumeration order does not identify camera roles.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
