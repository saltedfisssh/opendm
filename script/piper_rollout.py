#!/usr/bin/env python3
"""Run a cloud Piper policy on local cameras and pyAgxArm CAN drivers.

Streams action chunks continuously: a background thread keeps requesting new
chunks from the policy server while a fixed-rate control loop drains a
client-side buffer that drops the already-executed prefix of each new chunk
and crossfades the overlap (Real-Time Chunking, deployment-side only -- see
``docs/zh/piper_deployment.md``). This mirrors ``agilex_ros_infer.py``'s
layer-A RTC design, adapted from ROS/single-arm to pyAgxArm/dual-arm.

On Ctrl-C, SIGTERM, or any error, the script simply stops sending new
commands and disconnects; it never calls the SDK's electronic emergency
stop. Position-velocity motion (``move_j``/``move_p``) holds the last
commanded target on its own, so there is nothing else to do. A real
emergency needs the physical e-stop, not a driver call.
"""

import argparse
from contextlib import ExitStack
from collections import deque
import json
from pathlib import Path
import signal
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from opendm.deploy.piper import RUNGS, Representation, clip_gripper, finite_array
from opendm.data import se3
from opendm.kinematics import piper
from opendm.deploy.realsense import RealSenseCamera as Camera


class RtcActionBuffer:
    """Thread-safe queue with drop-prefix + temporal crossfade (client-side RTC).

    Ported from ``agilex_ros_infer.py``'s ``StreamActionBuffer``: the policy
    server is a stateless HTTP endpoint (no incremental/guided decoding), so
    RTC here is approximated client-side by dropping the stale prefix of each
    new chunk (the part already executed while the request was in flight) and
    crossfading it against the tail of the chunk still queued.
    """

    def __init__(self, min_smooth_steps=8, smooth_method="temporal", smooth_weight=1.0):
        self.min_smooth_steps = max(1, int(min_smooth_steps))
        self.smooth_method = str(smooth_method).lower()
        self.smooth_weight = float(min(max(smooth_weight, 0.0), 1.0))
        self.lock = threading.Lock()
        self.cur_chunk = deque()
        self.last_action = None

    def remaining(self):
        with self.lock:
            return len(self.cur_chunk)

    def integrate_new_chunk(self, actions_chunk, max_k, drop_n_override=None):
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            max_k = max(0, int(max_k))
            drop_n = max(0, min(int(drop_n_override), max_k)) if drop_n_override is not None else 0
            if drop_n >= len(actions_chunk):
                print(f"[rtc] drop_n={drop_n} >= chunk_len={len(actions_chunk)}, skip", flush=True)
                return
            new_list = [np.asarray(a, dtype=np.float64).copy() for a in actions_chunk[drop_n:]]

            if self.smooth_method == "raw":
                self.cur_chunk = deque(new_list)
                return

            min_m = self.min_smooth_steps
            if len(self.cur_chunk) == 0 and self.last_action is not None:
                old_list = [np.asarray(self.last_action, dtype=np.float64).copy() for _ in range(min_m)]
            else:
                old_list = [np.asarray(a, dtype=np.float64) for a in self.cur_chunk]
                if 0 < len(old_list) < min_m:
                    tail = old_list[-1].copy()
                    old_list.extend([tail.copy() for _ in range(min_m - len(old_list))])
                elif len(old_list) == 0:
                    self.cur_chunk = deque(new_list)
                    return

            overlap_len = min(len(old_list), len(new_list))
            if overlap_len <= 0:
                self.cur_chunk = deque(new_list)
                return
            old_list = old_list[:overlap_len]

            w_old = np.array([1.0]) if overlap_len == 1 else np.linspace(1.0, 0.0, overlap_len)
            w_old = self.smooth_weight * w_old
            w_new = 1.0 - w_old
            smoothed = [w_old[i] * old_list[i] + w_new[i] * new_list[i] for i in range(overlap_len)]
            self.cur_chunk = deque(a.copy() for a in smoothed + new_list[overlap_len:])

    def pop_next_action(self):
        with self.lock:
            if not self.cur_chunk:
                return None
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=np.float64).copy()
            return np.asarray(self.cur_chunk.popleft(), dtype=np.float64)


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
            # Let the firmware/SDK clamp joint targets to hardware limits
            # instead of a duplicate app-level check that aborts the whole
            # chunk on a small inference overshoot.
            arm.set_joint_limits_enabled(True)
            self.grippers.append(arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER))

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
                raise TimeoutError("Timed out waiting for live dual-arm feedback") from error
            halt.wait(0.02)
        raise RuntimeError("Stopped while waiting for arm feedback")

    def enable(self, speed, halt):
        deadline = time.monotonic() + 5
        while True:
            if halt.is_set():
                raise RuntimeError("Stopped before enabling")
            if all(arm.enable() for arm in self.arms):
                break
            if time.monotonic() > deadline:
                raise RuntimeError("Arm enable timed out")
            time.sleep(0.05)
        for arm in self.arms:
            arm.set_speed_percent(speed)

    def send(self, command, joint_mode, motion_mode, force):
        command = clip_gripper(finite_array(command, (14,)))
        for i, (arm, gripper) in enumerate(zip(self.arms, self.grippers)):
            values = command[i * 7 : i * 7 + 6].tolist()
            if joint_mode:
                mover = arm.move_js if motion_mode == "mit" else arm.move_j
                mover(values)
            else:
                arm.move_p(values)
            gripper.move_gripper_m(float(command[i * 7 + 6]), force=force)


def current_command(joints, joint_mode):
    if joint_mode:
        return joints.copy()
    result = joints.copy()
    for start in (0, 7):
        pose = piper.fk(joints[start : start + 6])
        result[start : start + 6] = np.r_[pose[:3, 3], se3.mat_to_rpy(pose[:3, :3])]
    return result


def infer_and_integrate(session, inference_url, rep, robot, cameras, args, buffer, state):
    """Run one inference round trip and fold the result into ``buffer``.

    Split out of the background inference loop so it can be driven
    synchronously, one call at a time, in tests -- without real threads or
    real wall-clock waits.
    """
    started = time.monotonic()
    commands = infer_once(
        session, inference_url, rep, robot, cameras, args.prompt, args.timeout, args.feedback_age
    )
    dt = 1.0 / rep.spec["fps"]
    if state["first_chunk"]:
        drop_n = 0
        state["first_chunk"] = False
    else:
        wall_steps = int(round((time.monotonic() - started) / dt))
        drop_n = max(0, min(wall_steps, args.rtc_latency_k))
    buffer.integrate_new_chunk(commands, max_k=args.rtc_latency_k, drop_n_override=drop_n)


def infer_once(session, inference_url, rep, robot, cameras, prompt, timeout, feedback_age):
    """One observe -> HTTP infer -> decode round trip; used by dry-run and RTC."""
    images = {str(i + 1): c.image(feedback_age) for i, c in enumerate(cameras)}
    joints = robot.feedback(feedback_age)
    state = rep.observe(joints)
    response = session.post(
        inference_url,
        json={
            "piper": rep.spec,
            "observation": {"prompt": prompt, "state": state.tolist(), "images": images},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("Server response must be a JSON object")
    metadata = body.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("piper") != rep.spec:
        raise ValueError("Server response representation contract mismatch")
    return rep.decode(state, body["actions"])


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
        help="Call the policy server once and print the action chunk without moving",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Enable arms and stream RTC commands; default only streams and prints",
    )
    parser.add_argument(
        "--motion-mode",
        choices=["smooth", "mit"],
        default="smooth",
        help=(
            "smooth = move_j/move_p (firmware position-velocity smoothing); "
            "mit = move_js (joint pass-through, no smoothing, lower latency). "
            "mit only applies to --rung s0 (true joint-space output); the SDK "
            "has no Cartesian pass-through equivalent for the EEF rungs."
        ),
    )
    parser.add_argument(
        "--max-steps", type=int, default=100_000, help="Stop after this many control ticks"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30,
        help="HTTP timeout and maximum observation age, seconds",
    )
    parser.add_argument("--feedback-age", type=float, default=0.5)
    parser.add_argument("--speed", type=int, default=10, help="SDK speed percent (smooth mode)")
    parser.add_argument("--force", type=float, default=1, help="Gripper force in N")
    parser.add_argument(
        "--rtc-latency-k",
        type=int,
        default=30,
        help="Max number of stale prefix steps to drop from a new chunk",
    )
    parser.add_argument(
        "--rtc-wait-steps",
        type=int,
        default=3,
        help="Start the next inference when this many queued steps remain",
    )
    parser.add_argument(
        "--rtc-min-smooth-steps",
        type=int,
        default=8,
        help="Minimum overlap length for temporal blending",
    )
    parser.add_argument(
        "--rtc-smooth-method",
        choices=["temporal", "raw"],
        default="temporal",
        help="temporal = crossfade overlap; raw = replace after drop_n",
    )
    parser.add_argument(
        "--rtc-smooth-weight",
        type=float,
        default=1.0,
        help="Old-chunk weight in the overlap (1=full blend, 0=pure new)",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    if not (
        args.max_steps >= 1
        and 0 < args.timeout <= 120
        and 0 < args.feedback_age <= 2
        and 1 <= args.speed <= 100
        and 0 < args.force <= 5
        and 0 < args.camera_startup_timeout <= 60
        and 0 <= args.camera_warmup_frames <= 300
        and args.rtc_latency_k >= 0
        and args.rtc_wait_steps >= 0
        and args.rtc_min_smooth_steps >= 1
        and 0 <= args.rtc_smooth_weight <= 1
    ):
        parser.error("Invalid max-steps/timeout/feedback-age/speed/force/camera/rtc settings")
    if args.left_can == args.right_can or len(set(args.cameras)) != 3:
        parser.error("Use distinct CAN interfaces and three distinct cameras")
    if args.motion_mode == "mit" and args.rung != "s0":
        parser.error("--motion-mode mit only applies to --rung s0")
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
            camera = Camera(source, args.camera_startup_timeout, args.camera_warmup_frames)
            stack.callback(camera.close)
            cameras.append(camera)
        # Validate all streams before connecting or enabling either arm.
        for camera in cameras:
            camera.check_fresh(args.feedback_age)
        robot = Robot([args.left_can, args.right_can], args.firmware, stack)
        joint_mode = args.rung == "s0"

        # Prime feedback freshness tracking before any enabling or inference.
        robot.wait_ready(args.feedback_age, 5.0, halt)
        for camera in cameras:
            camera.check_fresh(args.feedback_age)

        if args.dry_run:
            commands = infer_once(
                session, inference_url, rep, robot, cameras, args.prompt, args.timeout, args.feedback_age
            )
            print(
                json.dumps(
                    {
                        "steps": len(commands),
                        "first": commands[0].tolist(),
                        "last": commands[-1].tolist(),
                    }
                ),
                flush=True,
            )
            return

        if args.execute:
            robot.enable(args.speed, halt)

        buffer = RtcActionBuffer(
            min_smooth_steps=args.rtc_min_smooth_steps,
            smooth_method=args.rtc_smooth_method,
            smooth_weight=args.rtc_smooth_weight,
        )
        state = {"first_chunk": True}

        def inference_loop():
            # A single background thread issues inference requests
            # sequentially, so there is no concurrent-request state to track
            # beyond the buffer's own lock.
            while not halt.is_set():
                if buffer.remaining() > args.rtc_wait_steps:
                    halt.wait(0.005)
                    continue
                try:
                    infer_and_integrate(session, inference_url, rep, robot, cameras, args, buffer, state)
                except Exception as exc:
                    print(f"[rollout] infer failed: {exc}", flush=True)
                    halt.wait(0.2)

        worker = threading.Thread(target=inference_loop, name="piper-rtc-infer", daemon=True)
        worker.start()
        print(
            f"[rollout] rtc streaming  rung={args.rung}  motion_mode={args.motion_mode}  "
            f"execute={args.execute}  wait_steps={args.rtc_wait_steps}  latency_k={args.rtc_latency_k}",
            flush=True,
        )

        try:
            dt = 1.0 / rep.spec["fps"]
            step = 0
            last_action = None
            next_tick = time.monotonic()
            while step < args.max_steps and not halt.is_set():
                action = buffer.pop_next_action()
                if action is None:
                    action = last_action
                if action is not None:
                    last_action = action
                    if args.execute:
                        robot.send(action, joint_mode, args.motion_mode, args.force)
                step += 1
                if step % rep.spec["fps"] == 0:
                    print(
                        json.dumps({"step": step, "remain": buffer.remaining(), "execute": args.execute}),
                        flush=True,
                    )
                next_tick += dt
                halt.wait(max(0.0, next_tick - time.monotonic()))
        finally:
            halt.set()
            worker.join(timeout=1)


if __name__ == "__main__":
    main()
