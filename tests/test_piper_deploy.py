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


@pytest.mark.parametrize("rung", ["s1", "s2", "s2_pair", "s3", "s3a"])
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
    if rung in ("s2_pair", "s3"):
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


@pytest.mark.parametrize("rung", ["s2", "s2_pair", "s4"])
def test_server_contract_and_no_generic_absolute(monkeypatch, rung):
    from playground.dm05_piper import DM05InferenceConfig
    from opendm.exp.dm05_exp import DM05InferenceConfig as Base
    from flask import Flask

    config = DM05InferenceConfig(dataset_name=f"piper_fold_{rung}")
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
    # S3a also has 14-D state and 20-D actions; width alone is insufficient.
    assert client.post("/v1/infer", json={"piper": contract("s3a")}).status_code == 400
    assert client.post("/v1/infer", json={"piper": contract("s3")}).status_code == 400
    response = client.post("/v1/infer", json={"piper": contract(rung)})
    assert response.status_code == 200
    assert response.json["metadata"]["piper"] == contract(rung)
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


def make_fake_session(calls, failure, rung="s0", requests_log=None):
    import requests

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json, timeout):
            assert url == "http://cloud:7891/v1/infer"
            assert json["piper"] == contract(rung)
            assert len(json["observation"]["images"]) == 3
            if requests_log is not None:
                requests_log.append(json)
            if failure == "http":
                raise requests.Timeout("offline")
            actions = np.zeros((50, contract(rung)["action_dim"]))
            if actions.shape[1] == 20:
                actions[:, [3, 7, 13, 17]] = 1  # identity body rotations
                actions[:, [9, 19]] = 0.04
            else:
                actions[:, [6, 13]] = 0.04
            if failure == "nan":
                actions[0, 0] = np.nan
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "actions": actions.tolist(),
                    "metadata": {
                        "piper": contract("s1" if failure == "contract" else rung)
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


@pytest.mark.parametrize(
    "rung, options",
    [
        ("s0", []),
        ("s2_pair", []),
        ("s4", []),
        ("s4", ["--s4-episode-id", "cloth-003", "--s4-seed", "12", "--s4-xy-range", "0.2"]),
    ],
)
def test_dry_run_success_prints_chunk(monkeypatch, capsys, rung, options):
    import requests
    import sys

    module = load_rollout()
    calls = []
    requests_log = []

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
    monkeypatch.setattr(requests, "Session", make_fake_session(calls, "success", rung, requests_log))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "piper_rollout.py",
            "--server",
            "http://cloud:7891",
            "--rung",
            rung,
            "--cameras",
            "0",
            "1",
            "2",
            "--dry-run",
            *options,
        ],
    )
    module.main()
    assert "enable" not in calls
    import json
    from opendm.data.episode_frame import sample_episode_frame

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[-1]["steps"] == 50
    if rung == "s2_pair":
        observed = np.asarray(requests_log[0]["observation"]["state"])
        assert observed.shape == (23,)
        np.testing.assert_allclose(observed[:14], Representation("s2").observe(joints()))
        np.testing.assert_allclose(observed[14:], Representation("s3").observe(joints())[14:])
        physical = piper.fk(joints().reshape(2, 7)[:, :6])
        for key in ("first", "last"):
            command = np.asarray(records[-1][key]).reshape(2, 7)
            np.testing.assert_allclose(command[:, :3], physical[:, :3, 3], atol=1e-12)
            np.testing.assert_allclose(
                se3.rpy_to_mat(command[:, 3:6]), physical[:, :3, :3], atol=1e-12
            )
            np.testing.assert_allclose(command[:, 6], 0.04)
    if rung == "s4":
        metadata = records[0]["s4"]
        frame = sample_episode_frame(
            metadata["episode_id"], metadata["seed"], metadata["xy_range_m"]
        )
        np.testing.assert_array_equal(metadata["episode_frame"], frame)
        if options:
            assert (metadata["episode_id"], metadata["seed"], metadata["xy_range_m"]) == (
                "cloth-003", 12, 0.2
            )
        else:
            assert metadata["episode_id"]
            assert (metadata["seed"], metadata["xy_range_m"]) == (0, 0.5)
        # The request is in G, but returned commands must hold each real flange.
        observed = np.asarray(requests_log[0]["observation"]["state"]).reshape(2, 7)
        physical = piper.fk(joints().reshape(2, 7)[:, :6])
        expected = frame @ np.stack([np.eye(4), piper.T_RIGHT_BASE_TO_LEFT_BASE]) @ physical
        np.testing.assert_allclose(
            se3.pos_rotvec_to_transform(observed[:, :6]), expected, atol=1e-12
        )
        for key in ("first", "last"):
            command = np.asarray(records[-1][key]).reshape(2, 7)
            np.testing.assert_allclose(command[:, :3], physical[:, :3, 3], atol=1e-12)
            np.testing.assert_allclose(
                se3.rpy_to_mat(command[:, 3:6]), physical[:, :3, :3], atol=1e-12
            )
            np.testing.assert_allclose(command[:, 6], 0.04)


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


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_s4_matches_training_and_decodes_full_chunk_from_original_anchor(seed):
    from opendm.data.episode_frame import sample_episode_frame, transform_eef_states
    from opendm.data.transforms import ActionRelativeSE3
    from opendm.dataset.piper_dual import PIPER_EEF_STATE_DESC

    frame = sample_episode_frame("rollout", seed)
    rep = Representation("s4", episode_frame=frame)
    # Different trajectories for the two arms, plus absolute gripper targets.
    q = np.tile(joints(), (61, 1))
    t = np.linspace(0, 1, len(q))
    q[:, 0] += 0.4 * t
    q[:, 9] -= 0.2 * t
    q[:, 11] += 0.1 * t
    q[:, 6] -= 0.01 * t
    q[:, 13] += 0.02 * t
    physical = piper.fk(q.reshape(-1, 2, 7)[:, :, :6])
    world = rep.real_bases @ physical
    training = q.copy()
    for a in range(2):
        training[:, a * 7 : a * 7 + 6] = se3.transform_to_pos_rotvec(world[:, a])
    training = transform_eef_states(training, frame)
    observed = np.stack([rep.observe(row) for row in q])
    np.testing.assert_allclose(observed, training, atol=1e-12)
    np.testing.assert_array_equal(observed[:, [6, 13]], q[:, [6, 13]])

    # Use the actual training encoder, with a single t anchor for all 50 steps.
    # Decode after newer observations to catch accidental use of rep.previous.
    actions = ActionRelativeSE3()(
        {
            "state": training[0],
            "action": training[1:51][None],
            "meta_data": {"state_desc": PIPER_EEF_STATE_DESC},
        }
    )["action"][0]
    decoded = rep.decode(observed[0], actions).reshape(50, 2, 7)
    np.testing.assert_allclose(decoded[:, :, :3], physical[1:51, :, :3, 3], atol=2e-6)
    np.testing.assert_allclose(
        se3.rpy_to_mat(decoded[:, :, 3:6]), physical[1:51, :, :3, :3], atol=2e-6
    )
    np.testing.assert_allclose(decoded[:, :, 6], q[1:51, [6, 13]], atol=1e-7)


def test_s4_frame_is_copied_and_observation_rotation_is_unwrapped(monkeypatch):
    from opendm.data.episode_frame import transform_eef_states

    frame = se3.make_transform([0.2, -0.3, 0], se3.rotvec_to_mat([0, 0, 0.7]))
    rep = Representation("s4", episode_frame=frame)
    expected_frame = frame.copy()
    frame[:2, 3] += 10  # Caller changes must not change a running episode.
    np.testing.assert_array_equal(rep.episode_frame, expected_frame)
    assert not rep.episode_frame.flags.writeable
    local = se3.make_transform(
        np.tile([0.2, 0.1, 0.3], (3, 2, 1)),
        se3.rotvec_to_mat(
            np.array([[[0, 0, yaw]] * 2 for yaw in (2.4, 2.5, 2.6)])
        ),
    )
    poses = iter(local)
    monkeypatch.setattr(piper, "fk", lambda q: next(poses))
    observed = np.stack([rep.observe(joints()) for _ in local])
    world = rep.real_bases @ local
    states = np.tile(joints(), (len(local), 1))
    for a in range(2):
        states[:, a * 7 : a * 7 + 6] = se3.transform_to_pos_rotvec(world[:, a])
    np.testing.assert_allclose(
        observed, transform_eef_states(states, expected_frame), atol=1e-12
    )
    assert observed[-1, 5] > np.pi
    np.testing.assert_allclose(np.diff(observed[:, 5]), 0.1, atol=1e-12)
    # A new rollout starts with a fresh rotation chart.
    monkeypatch.setattr(piper, "fk", lambda q: local[-1])
    fresh = Representation("s4", episode_frame=expected_frame).observe(joints())
    assert fresh[5] < 0


@pytest.mark.parametrize(
    "frame",
    [
        None,
        np.eye(3),
        np.full((4, 4), np.nan),
        np.diag([2, 1, 1, 1]),  # scale
        np.diag([-1, 1, 1, 1]),  # reflection
        np.diag([1, -1, -1, 1]),  # opposite gravity sign
        np.diag([1, 1, 1, 2]),  # invalid homogeneous row
        se3.make_transform([0, 0, 0.1], np.eye(3)),  # height randomization
        se3.make_transform([0, 0, 0], se3.rotvec_to_mat([0.1, 0, 0])),  # roll
    ],
)
def test_s4_rejects_missing_or_incompatible_frame(frame):
    with pytest.raises(ValueError):
        Representation("s4", episode_frame=frame)


def test_episode_frame_rejected_for_other_rungs():
    with pytest.raises(ValueError, match="only applies to S4"):
        Representation("s3a", episode_frame=np.eye(4))


@pytest.mark.parametrize("rung", ["s4", "s2_pair"])
def test_server_contract_initializes_twenty_dimensional_output(monkeypatch, rung):
    from playground.dm05_piper import DM05Exp
    from opendm.exp.dm05_exp import DM05Exp as Base
    from opendm.dataset.piper_dual import PIPER_EEF_STATE_DESC, PIPER_EEF_PAIR_STATE_DESC

    exp = DM05Exp()
    exp.data_config.dataset_name = f"piper_fold_{rung}"
    monkeypatch.setattr(Base, "_initialize_inference_runtime", lambda self: None)
    with pytest.raises(ValueError, match="relative-mode se3"):
        exp._initialize_inference_runtime()
    exp.data_config.relative_mode = "se3"
    exp._initialize_inference_runtime()
    assert exp.inference_config.dataset_name == f"piper_fold_{rung}"
    assert exp.inference_config.output_action_dim == 20
    assert exp.inference_config._request_default_overrides()["default_state_desc"] == list(
        PIPER_EEF_PAIR_STATE_DESC if rung == "s2_pair" else PIPER_EEF_STATE_DESC
    )


@pytest.mark.parametrize(
    "rung, options",
    [
        ("s4", ["--s4-xy-range", "-1"]),
        ("s4", ["--s4-xy-range", "nan"]),
        ("s4", ["--s4-xy-range", "inf"]),
        ("s4", ["--s4-episode-id", " "]),
        ("s3a", ["--s4-seed", "0"]),
    ],
)
def test_bad_s4_cli_settings_fail_before_hardware(monkeypatch, rung, options):
    module = load_rollout()
    calls = []
    monkeypatch.setattr(module, "Camera", lambda *a: calls.append("camera"))
    monkeypatch.setattr(module, "Robot", lambda *a: calls.append("robot"))
    monkeypatch.setattr(
        "sys.argv",
        ["piper_rollout.py", "--server", "http://cloud", "--rung", rung, *options],
    )
    with pytest.raises(SystemExit):
        module.main()
    assert not calls


def test_s2_pair_matches_training_and_aux_does_not_change_decoded_targets():
    from script.piper_lerobot_to_jsonl import build_states
    from opendm.data.transforms import ActionRelativeSE3
    from opendm.dataset.piper_dual import PIPER_EEF_PAIR_STATE_DESC

    q = np.tile(joints(), (51, 1))
    t = np.linspace(0, 1, len(q))
    q[:, 0] += 0.3 * t
    q[:, 8] -= 0.1 * t
    q[:, 11] += 0.2 * t
    q[:, 6] -= 0.01 * t
    q[:, 13] += 0.02 * t
    rep = Representation("s2_pair")
    state = np.stack([rep.observe(row) for row in q])
    local = Representation("s2")
    shared = Representation("s3")
    np.testing.assert_allclose(state, build_states(q, "eef_local_pair"), atol=1e-12)
    np.testing.assert_array_equal(state[:, :14], np.stack([local.observe(row) for row in q]))
    np.testing.assert_allclose(
        state[:, 14:], np.stack([shared.observe(row)[14:] for row in q]), atol=1e-12
    )
    actions = ActionRelativeSE3()(
        {
            "state": state[0],
            "action": state[1:][None],
            "meta_data": {"state_desc": PIPER_EEF_PAIR_STATE_DESC},
        }
    )["action"][0]
    assert actions.shape == (50, 20)
    decoded = rep.decode(state[0], actions)
    changed_aux = state[0].copy()
    changed_aux[14:] = 123
    np.testing.assert_array_equal(rep.decode(changed_aux, actions), decoded)
    decoded = decoded.reshape(50, 2, 7)
    physical = piper.fk(q[1:].reshape(50, 2, 7)[:, :, :6])
    np.testing.assert_allclose(decoded[:, :, :3], physical[:, :, :3, 3], atol=2e-6)
    np.testing.assert_allclose(
        se3.rpy_to_mat(decoded[:, :, 3:6]), physical[:, :, :3, :3], atol=2e-6
    )
    np.testing.assert_allclose(decoded[:, :, 6], q[1:, [6, 13]], atol=1e-7)


@pytest.mark.parametrize("rung", ["s0", "s2", "s2_pair", "s3", "s4"])
def test_server_contract_uses_dataset_state_descriptor_and_validates_state(rung):
    import base64
    import io
    from flask import Flask
    from PIL import Image
    from playground.dm05_piper import DM05InferenceConfig, RUNG_STATE_DESCS

    image = io.BytesIO()
    Image.new("RGB", (8, 8)).save(image, format="PNG")
    encoded = base64.b64encode(image.getvalue()).decode()
    config = DM05InferenceConfig(dataset_name=f"piper_fold_{rung}")
    config.use_absolute_action = False
    config.is_history = False
    captured = []

    def predict(data):
        captured.append(data)
        return np.zeros((50, contract(rung)["action_dim"]))

    config._predict = predict
    config.last_model_latency_sec = None
    app = Flask(__name__)
    app.add_url_rule("/v1/infer", view_func=config._infer, methods=["POST"])
    client = app.test_client()
    body = {
        "piper": contract(rung),
        "observation": {
            "state": [0.0] * contract(rung)["state_dim"],
            "images": {str(i): encoded for i in (1, 2, 3)},
        },
    }
    assert client.post("/v1/infer", json=body).status_code == 200
    assert captured[-1]["meta_data"]["state_desc"] == list(
        RUNG_STATE_DESCS[f"piper_fold_{rung}"]
    )
    for bad in ([0.0], [0.0] * (contract(rung)["state_dim"] + 1), [float("nan")] * contract(rung)["state_dim"]):
        body["observation"]["state"] = bad
        assert client.post("/v1/infer", json=body).status_code == 400
    assert len(captured) == 1  # Bad observations never reach the model.
