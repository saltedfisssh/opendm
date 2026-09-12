import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from opendm.data import se3
from opendm.deploy.piper import Representation, clip_gripper, contract
from opendm.kinematics import piper


def joints():
    return np.r_[piper.neutral_joints(), 0.04, piper.neutral_joints(), 0.04]


@pytest.mark.parametrize("rung", ["s1", "s2", "s3", "s3a"])
def test_roundtrip_physical_target(rung):
    rep = Representation(rung)
    q = joints()
    state = rep.observe(q)
    target_q = q.copy()
    target_q[[0, 7]] += 0.02
    physical = piper.fk(target_q.reshape(2, 7)[:, :6])
    target = physical.copy()
    if rung in ("s3", "s3a"):
        target = rep.real_bases @ target
    blocks = []
    for a in range(2):
        anchor = se3.pos_rotvec_to_transform(state[a * 7 : a * 7 + 6])
        if rung == "s1":
            blocks.extend(
                np.r_[
                    target[a, :3, 3] - anchor[:3, 3],
                    se3.mat_to_rotvec(target[a, :3, :3] @ anchor[:3, :3].T),
                    0.03,
                ]
            )
        else:
            blocks.extend(
                np.r_[
                    se3.transform_to_pos_rot6d(
                        se3.transform_inverse(anchor) @ target[a]
                    ),
                    0.03,
                ]
            )
    decoded = rep.decode(state, [blocks]).reshape(2, 7)
    np.testing.assert_allclose(decoded[:, :3], physical[:, :3, 3], atol=1e-9)
    np.testing.assert_allclose(
        se3.rpy_to_mat(decoded[:, 3:6]), physical[:, :3, :3], atol=1e-9
    )
    np.testing.assert_allclose(decoded[:, 6], 0.03)
    if rung == "s3":
        np.testing.assert_allclose(
            state[14:],
            se3.transform_to_pos_rot6d(
                se3.transform_inverse(rep.real_bases[0] @ piper.fk(q[:6]))
                @ rep.real_bases[1]
                @ piper.fk(q[7:13])
            ),
            atol=1e-12,
        )


def test_joint_delta_gripper_absolute():
    rep = Representation("s0")
    q = joints()
    delta = np.full((2, 14), 0.01)
    out = rep.decode(q, delta)
    np.testing.assert_allclose(out[:, :6], np.tile(q[:6] + 0.01, (2, 1)))
    np.testing.assert_allclose(out[:, [6, 13]], 0.01)


def test_s5_uses_virtual_base_in_both_directions():
    bases = {
        "left": {"position": [0, 0, 0], "yaw_rad": 0.05},
        "right": {"position": [0, -0.6, 0], "yaw_rad": -0.05},
    }
    rep = Representation("s5", bases)
    q = joints()
    state = rep.observe(q)
    assert np.max(np.abs(state - q)) > 0.01
    delta = np.zeros((1, 14))
    delta[:, [6, 13]] = 0.04
    out = rep.decode(state, delta).reshape(2, 7)
    physical = piper.fk(q.reshape(2, 7)[:, :6])
    np.testing.assert_allclose(out[:, :3], physical[:, :3, 3], atol=0.003)
    np.testing.assert_allclose(
        se3.rpy_to_mat(out[:, 3:6]), physical[:, :3, :3], atol=0.03
    )


@pytest.mark.parametrize("actions", [[], [[0] * 14], [[float("nan")] * 20], [[0] * 20]])
def test_bad_se3_chunk_rejected(actions):
    rep = Representation("s2")
    with pytest.raises(ValueError):
        rep.decode(rep.observe(joints()), actions)


def test_clip_gripper_clamps_instead_of_rejecting():
    commands = np.tile(joints(), (3, 1))
    commands[0, 6] = 0.5  # overshoot past hardware max
    commands[1, 13] = -0.5  # undershoot past hardware min
    commands[2, [6, 13]] = [0.03, 0.05]  # already in range, untouched
    clipped = clip_gripper(commands)
    assert clipped[0, 6] == 0.08
    assert clipped[1, 13] == 0.0
    np.testing.assert_allclose(clipped[2, [6, 13]], [0.03, 0.05])
    # Non-gripper columns and the caller's array are left untouched.
    np.testing.assert_allclose(clipped[:, :6], commands[:, :6])
    assert commands[0, 6] == 0.5


def test_server_contract_and_no_generic_absolute(monkeypatch):
    from playground.dm05_piper import DM05InferenceConfig
    from opendm.exp.dm05_exp import DM05InferenceConfig as Base
    from flask import Flask

    config = DM05InferenceConfig(dataset_name="piper_fold_s2")
    captured = {}
    monkeypatch.setattr(Base, "_initialize", lambda self, **kw: captured.update(kw))
    config._initialize(use_absolute_action=True)
    assert captured["use_absolute_action"] is False
    monkeypatch.setattr(config, "_prepare_input", lambda body: body)
    monkeypatch.setattr(config, "_predict", lambda data: np.zeros((50, 20)))
    config.last_model_latency_sec = None
    app = Flask(__name__)
    app.add_url_rule("/v1/infer", view_func=config._infer, methods=["POST"])
    client = app.test_client()
    assert client.post("/v1/infer", json={"piper": contract("s0")}).status_code == 400
    response = client.post("/v1/infer", json={"piper": contract("s2")})
    assert response.status_code == 200
    assert response.json["metadata"]["piper"] == contract("s2")
    assert np.shape(response.json["actions"]) == (50, 20)


def load_rollout():
    spec = importlib.util.spec_from_file_location(
        "rollout_test", Path(__file__).parents[1] / "script/piper_rollout.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_send_dispatches_motion_mode_and_clips_gripper():
    module = load_rollout()
    robot = module.Robot.__new__(module.Robot)
    calls = []

    class Arm:
        def move_j(self, values):
            calls.append(("j", values))

        def move_js(self, values):
            calls.append(("js", values))

        def move_p(self, values):
            calls.append(("p", values))

    robot.arms = [Arm(), Arm()]
    robot.grippers = [
        SimpleNamespace(move_gripper_m=lambda value, force: calls.append(("g", value, force)))
        for _ in range(2)
    ]

    command = joints()
    command[6] = 0.5  # overshoot past hardware max -- must clip, not raise
    command[13] = -0.5  # undershoot past hardware min

    robot.send(command, True, "smooth", 1)
    assert [c[0] for c in calls if c[0] != "g"] == ["j", "j"]
    assert calls[1] == ("g", 0.08, 1)
    assert calls[3] == ("g", 0.0, 1)
    calls.clear()

    robot.send(command, True, "mit", 1)
    assert [c[0] for c in calls if c[0] != "g"] == ["js", "js"]
    calls.clear()

    robot.send(command, False, "smooth", 1)
    assert [c[0] for c in calls if c[0] != "g"] == ["p", "p"]


def test_invalid_right_arm_action_rejected_before_left_send():
    module = load_rollout()
    robot = module.Robot.__new__(module.Robot)
    calls = []
    robot.arms = [SimpleNamespace(move_j=lambda values: calls.append(values))] * 2
    robot.grippers = [SimpleNamespace(move_gripper_m=lambda *a, **kw: None)] * 2
    command = joints()
    command[7] = np.nan
    with pytest.raises(ValueError):
        robot.send(command, True, "smooth", 1)
    assert not calls


@pytest.mark.parametrize("frozen", [False, True])
def test_wait_ready_requires_feedback_progress(monkeypatch, frozen):
    module = load_rollout()
    robot = module.Robot.__new__(module.Robot)
    robot.stamps = {}
    tick = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: tick[0])

    class Halt:
        def is_set(self):
            return False

        def wait(self, seconds):
            tick[0] += seconds

    def feedback(age):
        if tick[0] == 0:
            raise RuntimeError("Feedback not received yet")
        for key in ("0:q", "0:g", "1:q", "1:g"):
            robot.stamps[key] = (1 if frozen and key == "1:g" else tick[0], tick[0])

    robot.feedback = feedback
    if frozen:
        with pytest.raises(TimeoutError, match="live dual-arm feedback"):
            robot.wait_ready(0.5, 0.1, Halt())
    else:
        robot.wait_ready(0.5, 0.1, Halt())


def test_rtc_buffer_raw_drop_and_replace():
    module = load_rollout()
    buf = module.RtcActionBuffer(smooth_method="raw")
    chunk = [np.full(3, float(i)) for i in range(5)]
    buf.integrate_new_chunk(chunk, max_k=10, drop_n_override=2)
    assert buf.remaining() == 3
    np.testing.assert_allclose(buf.pop_next_action(), [2, 2, 2])


def test_rtc_buffer_skip_when_drop_n_exceeds_chunk():
    module = load_rollout()
    buf = module.RtcActionBuffer(smooth_method="raw")
    buf.integrate_new_chunk([np.zeros(3)] * 3, max_k=10, drop_n_override=0)
    assert buf.remaining() == 3
    # A new chunk whose entire prefix is already stale is skipped outright
    # rather than emptying the buffer -- the old queued actions keep playing.
    buf.integrate_new_chunk([np.ones(3)] * 2, max_k=10, drop_n_override=5)
    assert buf.remaining() == 3
    np.testing.assert_allclose(buf.pop_next_action(), [0, 0, 0])


def test_rtc_buffer_temporal_crossfade_blends_overlap():
    module = load_rollout()
    buf = module.RtcActionBuffer(min_smooth_steps=1, smooth_method="temporal", smooth_weight=1.0)
    buf.integrate_new_chunk([np.array([0.0])] * 3, max_k=10, drop_n_override=0)
    buf.integrate_new_chunk([np.array([10.0])] * 3, max_k=10, drop_n_override=0)
    blended = [buf.pop_next_action()[0] for _ in range(3)]
    assert blended[0] == pytest.approx(0.0)
    assert blended[2] == pytest.approx(10.0)
    assert 0.0 < blended[1] < 10.0


def test_rtc_buffer_pads_short_old_chunk_before_blend():
    module = load_rollout()
    buf = module.RtcActionBuffer(min_smooth_steps=4, smooth_method="temporal", smooth_weight=1.0)
    buf.integrate_new_chunk([np.array([1.0]), np.array([1.0])], max_k=10, drop_n_override=0)
    assert buf.remaining() == 2
    buf.integrate_new_chunk([np.array([9.0])] * 4, max_k=10, drop_n_override=0)
    assert buf.remaining() == 4


def test_infer_and_integrate_drops_prefix_by_elapsed_time(monkeypatch):
    module = load_rollout()
    rep = Representation("s0")
    robot = SimpleNamespace(feedback=lambda age: joints())
    cameras = [SimpleNamespace(image=lambda age: "jpeg") for _ in range(3)]
    actions = np.zeros((10, 14))
    actions[:, [6, 13]] = 0.04

    class Session:
        def post(self, url, json, timeout):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"actions": actions.tolist(), "metadata": {"piper": rep.spec}},
            )

    args = SimpleNamespace(prompt="p", timeout=1, feedback_age=0.5, rtc_latency_k=5)
    buffer = module.RtcActionBuffer(smooth_method="raw")
    state = {"first_chunk": True}

    # First chunk always keeps drop_n at 0 even if "elapsed" looks large.
    ticks = iter([0.0, 5.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks, 5.0))
    module.infer_and_integrate(Session(), "http://x/v1/infer", rep, robot, cameras, args, buffer, state)
    assert buffer.remaining() == 10
    assert state["first_chunk"] is False

    # Later chunks drop a prefix proportional to elapsed wall time, capped
    # at rtc_latency_k.
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks, 1.0))
    module.infer_and_integrate(Session(), "http://x/v1/infer", rep, robot, cameras, args, buffer, state)
    assert buffer.remaining() == 10 - 5


def test_deployment_defaults_and_firmware_alias():
    parser = load_rollout().build_argparser()
    args = parser.parse_args(["--server", "http://cloud:7891", "--rung", "s0"])
    assert (args.left_can, args.right_can) == ("can_l_slave", "can_r_slave")
    assert args.cameras == ["346522076596", "346522072780", "346522075577"]
    assert args.firmware == "v189"
    assert not args.execute
    assert args.motion_mode == "smooth"
    assert (
        parser.parse_args(
            ["--server", "http://cloud", "--rung", "s0", "--firmware", "V189"]
        ).firmware
        == "v189"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--server", "http://cloud", "--rung", "s0", "--execute", "--dry-run"]
        )


def test_mit_motion_mode_requires_joint_rung():
    module = load_rollout()
    import sys

    argv = [
        "piper_rollout.py",
        "--server",
        "http://cloud:7891",
        "--rung",
        "s2",
        "--motion-mode",
        "mit",
    ]
    orig_argv = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit):
            module.main()
    finally:
        sys.argv = orig_argv


class FakeCamera:
    def __init__(self, *args):
        self.checks = 0

    def check_fresh(self, *args):
        self.checks += 1

    def image(self, *args):
        return "jpeg"

    def close(self):
        pass


def make_fake_session(calls, failure):
    import requests

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json, timeout):
            assert url == "http://cloud:7891/v1/infer"
            assert json["piper"] == contract("s0")
            assert len(json["observation"]["images"]) == 3
            if failure == "http":
                raise requests.Timeout("offline")
            actions = np.zeros((50, 14))
            actions[:, [6, 13]] = 0.04
            if failure == "nan":
                actions[0, 0] = np.nan
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "actions": actions.tolist(),
                    "metadata": {
                        "piper": contract("s1" if failure == "contract" else "s0")
                    },
                },
            )

    return Session


@pytest.mark.parametrize("failure", ["http", "contract", "nan"])
def test_dry_run_infer_errors_propagate(monkeypatch, failure):
    import requests
    import sys

    module = load_rollout()
    calls = []

    class Robot:
        def __init__(self, *args):
            pass

        def feedback(self, *args):
            return joints()

        def wait_ready(self, *args):
            pass

        def enable(self, *args):
            calls.append("enable")

    monkeypatch.setattr(module, "Robot", Robot)
    monkeypatch.setattr(module, "Camera", FakeCamera)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(requests, "Session", make_fake_session(calls, failure))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "piper_rollout.py",
            "--server",
            "http://cloud:7891",
            "--rung",
            "s0",
            "--cameras",
            "0",
            "1",
            "2",
            "--dry-run",
        ],
    )
    with pytest.raises((requests.Timeout, ValueError)):
        module.main()
    assert "enable" not in calls


def test_dry_run_success_prints_chunk(monkeypatch, capsys):
    import requests
    import sys

    module = load_rollout()
    calls = []

    class Robot:
        def __init__(self, *args):
            pass

        def feedback(self, *args):
            return joints()

        def wait_ready(self, *args):
            pass

        def enable(self, *args):
            calls.append("enable")

    monkeypatch.setattr(module, "Robot", Robot)
    monkeypatch.setattr(module, "Camera", FakeCamera)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(requests, "Session", make_fake_session(calls, "success"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "piper_rollout.py",
            "--server",
            "http://cloud:7891",
            "--rung",
            "s0",
            "--cameras",
            "0",
            "1",
            "2",
            "--dry-run",
        ],
    )
    module.main()
    assert "enable" not in calls
    assert '"steps": 50' in capsys.readouterr().out


def test_camera_failure_before_playback(monkeypatch):
    import requests
    import sys

    module = load_rollout()
    calls = []

    class Camera:
        def __init__(self, *args):
            self.checks = 0

        def check_fresh(self, *args):
            self.checks += 1
            if self.checks == 2:
                raise RuntimeError("RealSense disconnected before playback")

        def image(self, *args):
            return "jpeg"

        def close(self):
            calls.append("camera_close")

    class Robot:
        def __init__(self, *args):
            pass

        def feedback(self, *args):
            return joints()

        def wait_ready(self, *args):
            pass

        def enable(self, *args):
            calls.append("enable")

    monkeypatch.setattr(module, "Robot", Robot)
    monkeypatch.setattr(module, "Camera", Camera)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(requests, "Session", make_fake_session(calls, "success"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "piper_rollout.py",
            "--server",
            "http://cloud:7891",
            "--rung",
            "s0",
            "--cameras",
            "0",
            "1",
            "2",
            "--execute",
        ],
    )
    with pytest.raises(RuntimeError, match="RealSense disconnected"):
        module.main()
    assert "enable" not in calls
    assert calls.count("camera_close") == 3


def test_execute_rtc_streams_and_sends(monkeypatch):
    import requests
    import sys

    module = load_rollout()
    calls = []

    class Robot:
        def __init__(self, *args):
            pass

        def feedback(self, *args):
            return joints()

        def wait_ready(self, *args):
            pass

        def enable(self, *args):
            calls.append("enable")

        def send(self, *args):
            calls.append("send")

    class Camera:
        def __init__(self, *args):
            self.checks = 0

        def check_fresh(self, *args):
            self.checks += 1

        def image(self, *args):
            return "jpeg"

        def close(self):
            calls.append("camera_close")

    monkeypatch.setattr(module, "Robot", Robot)
    monkeypatch.setattr(module, "Camera", Camera)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(requests, "Session", make_fake_session(calls, "success"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "piper_rollout.py",
            "--server",
            "http://cloud:7891",
            "--rung",
            "s0",
            "--cameras",
            "0",
            "1",
            "2",
            "--execute",
            # ~1s of real wall clock at 30 fps; the fake HTTP call is
            # instant, so the background inference thread has ample time
            # to deliver at least one chunk for the control loop to send.
            "--max-steps",
            "30",
        ],
    )
    module.main()
    assert "enable" in calls
    assert calls.count("send") >= 1
    assert calls.count("camera_close") == 3
