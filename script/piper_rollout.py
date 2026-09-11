#!/usr/bin/env python3
"""Run a cloud Piper policy on local cameras and pyAgxArm CAN drivers."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import signal
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from opendm.deploy.piper import RUNGS, Representation, finite_array, validate_motion
from opendm.data import se3
from opendm.kinematics import piper
from opendm.deploy.realsense import RealSenseCamera as Camera


class Robot:
    def __init__(self, channels, firmware, stack):
        from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

        self.arms = []
        self.grippers = []
        self.stamps = {}
        for channel in channels:
            arm = AgxArmFactory.create_arm(
                create_agx_arm_config(
                    robot=ArmModel.PIPER,
                    channel=channel,
                    interface="socketcan",
                    firmeware_version=getattr(PiperFW, firmware.upper()),
                )
            )
            self.arms.append(arm)
            stack.callback(arm.disconnect)
            arm.connect()
            self.grippers.append(arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER))

    def stop(self):
        # Attempt both even if one CAN adapter has failed.
        for arm in self.arms:
            try:
                arm.electronic_emergency_stop()
            except Exception as exc:
                print(f"E-stop failed: {exc}", file=sys.stderr)

    def feedback(self, max_age):
        result = []
        for i, (arm, gripper) in enumerate(zip(self.arms, self.grippers)):
            if not arm.is_ok():
                raise RuntimeError(f"Arm {i} communication unavailable")
            q, g = arm.get_joint_angles(), gripper.get_gripper_status()
            for key, message in ((f"{i}:q", q), (f"{i}:g", g)):
                if message is None or message.timestamp <= 0:
                    raise RuntimeError(f"Missing feedback: {key}")
                # Track progress using a local monotonic clock; SDK timestamp
                # origins differ between CAN backends.
                old, changed = self.stamps.get(key, (None, time.monotonic()))
                if old != message.timestamp:
                    changed = time.monotonic()
                self.stamps[key] = (message.timestamp, changed)
                if time.monotonic() - changed > max_age:
                    raise RuntimeError(f"Stale feedback: {key}")
            if g.msg.mode != "width":
                raise RuntimeError("Gripper feedback must use width mode")
            angles = finite_array(q.msg, (6,))
            width = float(g.msg.value)
            if not np.isfinite(width) or not 0 <= width <= 0.08:
                raise RuntimeError(f"Invalid gripper width on arm {i}: {width}")
            result.extend([*angles, width])
        return np.asarray(result)

    def wait_ready(self, max_age, timeout, halt):
        """Wait for asynchronous feedback and require both arms to update."""
        deadline = time.monotonic() + timeout
        initial = None
        error = None
        while not halt.is_set():
            try:
                self.feedback(max_age)
                stamps = {key: value[0] for key, value in self.stamps.items()}
                if initial is None:
                    initial = stamps
                elif all(stamps[key] != stamp for key, stamp in initial.items()):
                    return
            except (RuntimeError, ValueError) as exc:
                error = exc
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for live dual-arm feedback"
                ) from error
            halt.wait(0.02)
        raise RuntimeError("Stopped while waiting for arm feedback")

    def enable(self, speed, halt, lock):
        deadline = time.monotonic() + 5
        while True:
            with lock:
                if halt.is_set():
                    raise RuntimeError("Stopped before enabling")
                enabled = [arm.enable() for arm in self.arms]
            if all(enabled):
                break
            if time.monotonic() > deadline:
                raise RuntimeError("Arm enable timed out")
            time.sleep(0.05)
        for arm in self.arms:
            arm.set_speed_percent(speed)

    def send(self, command, joint_mode, force):
        command = finite_array(command, (14,))
        for i, (arm, gripper) in enumerate(zip(self.arms, self.grippers)):
            values = command[i * 7 : i * 7 + 6].tolist()
            (arm.move_j if joint_mode else arm.move_p)(values)
            gripper.move_gripper_m(float(command[i * 7 + 6]), force=force)


def current_command(joints, joint_mode):
    if joint_mode:
        return joints.copy()
    result = joints.copy()
    for start in (0, 7):
        pose = piper.fk(joints[start : start + 6])
        result[start : start + 6] = np.r_[pose[:3, 3], se3.mat_to_rpy(pose[:3, :3])]
    return result


def build_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, help="http://IP:port (or HTTPS URL)")
    parser.add_argument("--rung", choices=RUNGS, required=True)
    parser.add_argument("--left-can", default="can_l_slave")
    parser.add_argument("--right-can", default="can_r_slave")
    parser.add_argument(
        "--firmware",
        type=str.lower,
        default="v189",
        choices=["default", "v183", "v188", "v189"],
    )
    parser.add_argument(
        "--cameras",
        nargs=3,
        default=["346522076596", "346522072780", "346522075577"],
        metavar=("HEAD_SERIAL", "LEFT_SERIAL", "RIGHT_SERIAL"),
        help="RealSense serial numbers in Head / Left wrist / Right wrist order",
    )
    parser.add_argument(
        "--camera-startup-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for color frames after starting each camera",
    )
    parser.add_argument(
        "--camera-warmup-frames",
        type=int,
        default=15,
        help="Discard this many color frames per camera before use",
    )
    parser.add_argument("--bases", type=Path, help="S5 training estimated_bases.json")
    parser.add_argument(
        "--prompt",
        default="Task: fold the cloth. Scene: internal. Type: teleop. Quality: 5.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Observe and infer without enabling or moving; requires devices",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Enable arms and send commands; default observes only",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Execute this many chunk steps before reobserving",
    )
    parser.add_argument(
        "--cycles", type=int, default=1, help="Number of requests; 0 runs until Ctrl-C"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30,
        help="HTTP timeout and maximum observation age, seconds",
    )
    parser.add_argument("--feedback-age", type=float, default=0.5)
    parser.add_argument("--speed", type=int, default=10, help="SDK speed percent")
    parser.add_argument("--force", type=float, default=1, help="Gripper force in N")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    if not (
        1 <= args.steps <= 50
        and args.cycles >= 0
        and 0 < args.timeout <= 120
        and 0 < args.feedback_age <= 2
        and 1 <= args.speed <= 100
        and 0 < args.force <= 5
        and 0 < args.camera_startup_timeout <= 60
        and 0 <= args.camera_warmup_frames <= 300
    ):
        parser.error(
            "Invalid steps/cycles/timeout/feedback-age/speed/force/camera settings"
        )
    if args.left_can == args.right_can or len(set(args.cameras)) != 3:
        parser.error("Use distinct CAN interfaces and three distinct cameras")
    from urllib.parse import urlsplit

    endpoint = urlsplit(args.server)
    if endpoint.scheme not in ("http", "https") or not endpoint.netloc:
        parser.error("--server must be an HTTP(S) URL")
    if (
        endpoint.path.rstrip("/") not in ("", "/v1/infer")
        or endpoint.query
        or endpoint.fragment
    ):
        parser.error("--server must be a base URL or a full /v1/infer URL")
    inference_url = args.server.rstrip("/")
    if not inference_url.endswith("/v1/infer"):
        inference_url += "/v1/infer"
    rep = Representation(
        args.rung, json.loads(args.bases.read_text()) if args.bases else None
    )
    import requests

    halt = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: halt.set())
    signal.signal(signal.SIGTERM, lambda *_: halt.set())
    with ExitStack() as stack:
        session = stack.enter_context(requests.Session())
        cameras = []
        for source in args.cameras:
            camera = Camera(
                source, args.camera_startup_timeout, args.camera_warmup_frames
            )
            stack.callback(camera.close)
            cameras.append(camera)
        # Validate all streams before connecting or enabling either arm.
        for camera in cameras:
            camera.check_fresh(args.feedback_age)
        robot = Robot([args.left_can, args.right_can], args.firmware, stack)
        lock = threading.Lock()
        deadline = [time.monotonic() + 10]
        watchdog_done = threading.Event()

        def watchdog():
            while not watchdog_done.wait(0.02):
                with lock:
                    if halt.is_set() or time.monotonic() > deadline[0]:
                        halt.set()
                        if args.execute:
                            robot.stop()
                        return

        worker = threading.Thread(target=watchdog, daemon=True)
        worker.start()
        try:
            # Prime feedback freshness tracking before any enabling or inference.
            robot.wait_ready(args.feedback_age, 5.0, halt)
            for camera in cameras:
                camera.check_fresh(args.feedback_age)
            if args.execute:
                robot.enable(args.speed, halt, lock)
            cycle = 0
            while not halt.is_set() and (args.cycles == 0 or cycle < args.cycles):
                started = time.monotonic()
                with lock:
                    deadline[0] = started + args.timeout
                images = {
                    str(i + 1): c.image(args.feedback_age)
                    for i, c in enumerate(cameras)
                }
                joints = robot.feedback(args.feedback_age)
                state = rep.observe(joints)
                response = session.post(
                    inference_url,
                    json={
                        "piper": rep.spec,
                        "observation": {
                            "prompt": args.prompt,
                            "state": state.tolist(),
                            "images": images,
                        },
                    },
                    timeout=args.timeout,
                )
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise ValueError("Server response must be a JSON object")
                metadata = body.get("metadata")
                if not isinstance(metadata, dict) or metadata.get("piper") != rep.spec:
                    raise ValueError("Server response representation contract mismatch")
                commands = rep.decode(state, body["actions"])
                if len(commands) < args.steps:
                    raise ValueError("Server chunk shorter than --steps")
                commands = commands[: args.steps]
                if halt.is_set() or time.monotonic() - started > args.timeout:
                    raise TimeoutError("Inference result is stale")
                joint_mode = args.rung == "s0"
                current = current_command(robot.feedback(args.feedback_age), joint_mode)
                validate_motion(commands, current, joint_mode)
                print(
                    json.dumps(
                        {
                            "cycle": cycle,
                            "latency_s": time.monotonic() - started,
                            "execute": args.execute,
                            "commands": commands.tolist(),
                        }
                    ),
                    flush=True,
                )
                playback = time.monotonic()
                for index, command in enumerate(commands):
                    with lock:
                        if halt.is_set():
                            raise RuntimeError("Rollout stopped by watchdog or signal")
                        deadline[0] = time.monotonic() + args.feedback_age
                        current = current_command(
                            robot.feedback(args.feedback_age), joint_mode
                        )
                        validate_motion(command[None], current, joint_mode)
                        for camera in cameras:
                            camera.check_fresh(args.feedback_age)
                        if args.execute:
                            robot.send(command, joint_mode, args.force)
                    halt.wait(max(0, playback + (index + 1) / 30 - time.monotonic()))
                cycle += 1
        finally:
            halt.set()
            if args.execute:
                with lock:
                    robot.stop()
            watchdog_done.set()
            worker.join(timeout=1)


if __name__ == "__main__":
    main()
