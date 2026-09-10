import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from opendm.data import se3
from opendm.deploy.piper import Representation, contract, validate_motion
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


def test_entire_prefix_validated():
    q = joints()
    commands = np.tile(q, (5, 1))
    commands[-1, 0] += 1
    with pytest.raises(ValueError, match="Joint step"):
        validate_motion(commands, q, True)
    commands[-1] = q
    validate_motion(commands, q, True)
    commands[-1, 6] = -0.001
    with pytest.raises(ValueError, match="Gripper width"):
        validate_motion(commands, q, True)


def test_euler_crossing_rejected_even_for_small_physical_rotation():
    q = np.zeros(14)
    q[3] = np.pi - 0.01
    target = q.copy()
    target[3] = -np.pi + 0.01
    with pytest.raises(ValueError, match="Euler branch"):
        validate_motion(target[None], q, False)


def test_sdk_si_units_and_both_arm_stop():
    spec = importlib.util.spec_from_file_location(
        "rollout", Path(__file__).parents[1] / "script/piper_rollout.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    robot = module.Robot.__new__(module.Robot)
    calls = []

    class Arm:
        def move_j(self, values):
            calls.append(("j", values))

        def move_p(self, values):
            calls.append(("p", values))

        def electronic_emergency_stop(self):
            calls.append(("stop",))
            raise RuntimeError("CAN failure")

    robot.arms = [Arm(), Arm()]
    robot.grippers = [
        SimpleNamespace(
            move_gripper_m=lambda value, force: calls.append(("g", value, force))
        )
    ] * 2
    robot.send(joints(), True, 1)
    assert calls[1] == ("g", 0.04, 1)
    np.testing.assert_allclose(calls[0][1], joints()[:6])
    robot.stop()
    assert calls[-2:] == [("stop",), ("stop",)]


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


@pytest.mark.parametrize(
    "failure", ["http", "contract", "nan", "stale", "camera", "success", "dry_run"]
)
def test_rollout_failure_stops_without_sending(monkeypatch, failure):
    import requests
    import sys

    spec = importlib.util.spec_from_file_location(
        "rollout_test", Path(__file__).parents[1] / "script/piper_rollout.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    class Robot:
        def __init__(self, *args):
            pass

        def feedback(self, *args):
            return joints()

        def enable(self, *args):
            calls.append("enable")

        def stop(self):
            calls.append("stop")

        def send(self, *args):
            calls.append("send")

    class Camera:
        def __init__(self, *args):
            self.checks = 0

        def check_fresh(self, *args):
            self.checks += 1
            if failure == "camera" and self.checks == 3:
                raise RuntimeError("RealSense disconnected before playback")

        def image(self, *args):
            return "jpeg"

        def close(self):
            calls.append("camera_close")

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
            if failure == "stale":
                module.threading.Event().wait(0.1)
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

    monkeypatch.setattr(module, "Robot", Robot)
    monkeypatch.setattr(module, "Camera", Camera)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(requests, "Session", Session)
    argv = [
        "piper_rollout.py",
        "--server",
        "http://cloud:7891",
        "--rung",
        "s0",
        "--cameras",
        "0",
        "1",
        "2",
        "--steps",
        "1",
    ]
    if failure != "dry_run":
        argv += ["--execute"]
    if failure == "stale":
        argv += ["--timeout", "0.03"]
    monkeypatch.setattr(sys, "argv", argv)
    if failure in ("success", "dry_run"):
        module.main()
    else:
        with pytest.raises((requests.Timeout, ValueError, TimeoutError, RuntimeError)):
            module.main()
    if failure == "dry_run":
        assert "enable" not in calls and "stop" not in calls
    else:
        assert "stop" in calls
    assert calls.count("send") == (1 if failure == "success" else 0)
    assert calls.count("camera_close") == 3
