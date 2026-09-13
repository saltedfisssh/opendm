"""Regression checks for episode frames, split isolation and the installed FK."""

import json

import numpy as np
import pytest

from opendm.data import se3
from opendm.data.episode_frame import sample_episode_frame, transform_eef_states
from opendm.data.transforms import ActionRelativeSE3
from opendm.dataset.piper_dual import PIPER_EEF_STATE_DESC
from opendm.eval.piper_common_space import (
    decode_to_common_space,
    ground_truth_common_space,
)
from opendm.kinematics.robotwin2 import RoboTwinKinematics, ROBOTWIN_ROOT
from script.robotwin2_prepare_ablation import main, build_states


def test_s4_preserves_actions_pairing_grippers_and_decodes_to_piper_base():
    rng = np.random.default_rng(12)
    states = rng.normal(scale=0.2, size=(8, 14))
    h = sample_episode_frame("task/clean/episode0.jsonl", 0)
    randomized = transform_eef_states(states, h)
    np.testing.assert_array_equal(
        h, sample_episode_frame("task/clean/episode0.jsonl", 0)
    )
    assert not np.allclose(h, sample_episode_frame("task/randomized/episode0.jsonl", 0))
    assert not np.allclose(states[:, :3], randomized[:, :3])
    np.testing.assert_allclose(h[:3, 2], [0, 0, 1])
    assert h[2, 3] == 0
    np.testing.assert_array_equal(states[:, [6, 13]], randomized[:, [6, 13]])

    def encode(x):
        return ActionRelativeSE3()(
            {
                "state": x[0],
                "action": x[1:][None],
                "meta_data": {"state_desc": PIPER_EEF_STATE_DESC},
            }
        )["action"][0]

    action = encode(randomized)
    np.testing.assert_allclose(action, encode(states), atol=2e-6)
    restored, grippers = decode_to_common_space(
        "piper_fold_s4", randomized[0], action, PIPER_EEF_STATE_DESC, episode_frame=h
    )
    truth, truth_grippers = ground_truth_common_space(
        states[1:], PIPER_EEF_STATE_DESC, True
    )
    np.testing.assert_allclose(restored, truth, atol=2e-6)
    np.testing.assert_allclose(grippers, truth_grippers, atol=1e-7)
    with pytest.raises(ValueError, match="episode_frame"):
        decode_to_common_space(
            "piper_fold_s4", randomized[0], action, PIPER_EEF_STATE_DESC
        )
    for x in (states, randomized):
        pair = se3.relative_transform(
            se3.pos_rotvec_to_transform(x[:, :6]),
            se3.pos_rotvec_to_transform(x[:, 7:13]),
        )
        if x is states:
            original_pair = pair
        else:
            np.testing.assert_allclose(pair, original_pair, atol=1e-12)


def test_split_is_full_before_limit_and_resume_rejects_changes(tmp_path):
    source = tmp_path / "source"
    for setting in ("clean", "randomized"):
        directory = source / "jsonl/task" / setting
        directory.mkdir(parents=True)
        for i in range(4):
            rows = [
                {
                    "state": [float(i)] * 14,
                    "prompt": "test",
                    "images_1": {"url": f"./task/{setting}/episode{i}/cam_high.mp4"},
                }
            ] * 2
            (directory / f"episode{i}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
    dest = tmp_path / "output"
    args = [
        "--source",
        str(source),
        "--out-root",
        str(dest),
        "--rung",
        "s0",
        "--workers",
        "1",
        "--assets",
        str(tmp_path / "missing-assets"),
    ]
    main(args + ["--limit", "1"])
    split = json.loads((dest / "split.json").read_text())
    assert len(split["train"]) == 6 and len(split["held_out"]) == 2
    assert not set(split["train"]) & set(split["held_out"])
    main(args)
    assert json.loads((dest / "split.json").read_text()) == split
    assert len(list((dest / "joint/jsonl").rglob("*.jsonl"))) == 8
    with pytest.raises(ValueError, match="configuration/source inventory changed"):
        main(args + ["--seed", "1"])
    episode = source / "jsonl/task/clean/episode0.jsonl"
    episode.write_text(episode.read_text().replace("test", "changed"))
    with pytest.raises(ValueError, match="source changed"):
        main(args)


def test_robotwin_unified_fk_and_aux():
    if not (ROBOTWIN_ROOT / "assets/embodiments/aloha-agilex/config.yml").exists():
        pytest.skip("RoboTwin assets unavailable")
    kin = RoboTwinKinematics()
    state = np.random.default_rng(6).normal(scale=0.2, size=(9, 14))
    local = build_states(state, kin, "eef_local")
    unified = build_states(state, kin, "eef_unified_pair")
    right_base_to_left = (
        se3.transform_inverse(kin.root_to_bases[0]) @ kin.root_to_bases[1]
    )
    np.testing.assert_allclose(
        se3.pos_rotvec_to_transform(unified[:, 7:13]),
        right_base_to_left @ se3.pos_rotvec_to_transform(local[:, 7:13]),
        atol=1e-12,
    )
    assert unified.shape == (9, 23)
    from opendm.constants.robot import RobotStateDesc

    encoded = ActionRelativeSE3()(
        {
            "state": unified[0],
            "action": unified[1:][None],
            "meta_data": {
                "state_desc": PIPER_EEF_STATE_DESC + [RobotStateDesc.AUX] * 9
            },
        }
    )
    assert encoded["action"].shape == (1, 8, 20)


def test_piper_s4_cli_preserves_split_and_can_resume(tmp_path, monkeypatch):
    from script.piper_prepare_s4 import main as prepare_piper

    (tmp_path / "split.json").write_text(json.dumps({"train": [0], "held_out": [1]}))
    states = np.random.default_rng(1).normal(scale=0.1, size=(4, 14))
    for split, index in [("train", 0), ("held_out", 1)]:
        path = tmp_path / "eef_unified/jsonl" / split / f"episode_{index:06d}.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "state": s.tolist(),
                        "prompt": "fold",
                        "images_1": {"url": "original.mp4"},
                    }
                )
                + "\n"
                for s in states
            )
        )
    monkeypatch.setattr("sys.argv", ["piper_prepare_s4", "--data-root", str(tmp_path)])
    prepare_piper()
    output = tmp_path / "eef_gravity_random"
    frames = json.loads((output / "episode_frames.json").read_text())
    before = (output / "jsonl/train/episode_000000.jsonl").read_bytes()
    prepare_piper()
    assert (output / "jsonl/train/episode_000000.jsonl").read_bytes() == before
    for split, index in [("train", 0), ("held_out", 1)]:
        name = f"episode_{index:06d}.jsonl"
        rows = [
            json.loads(line)
            for line in (output / "jsonl" / split / name).read_text().splitlines()
        ]
        assert all(
            r["images_1"]["url"] == "original.mp4" and "action" not in r for r in rows
        )
        restored = transform_eef_states(
            np.array([r["state"] for r in rows]),
            se3.transform_inverse(np.asarray(frames[name])),
        )
        np.testing.assert_allclose(restored, states, atol=1e-12)


def test_offline_terminal_frame_repeats_last_state(tmp_path):
    from types import SimpleNamespace
    from script.piper_eval_offline import read_absolute_states

    path = tmp_path / "episode.jsonl"
    state = np.arange(14, dtype=float)
    path.write_text(json.dumps({"state": state.tolist()}) + "\n")
    dataset = SimpleNamespace(sample_index=[(0, 0)], id_to_jsonl={0: str(path)})
    anchor, future = read_absolute_states(dataset, 0, 3)
    np.testing.assert_array_equal(anchor, state)
    np.testing.assert_array_equal(future, np.repeat(state[None], 3, axis=0))


def test_large_process_pool_backlog_finishes_without_wakeup_pipe_deadlock(tmp_path):
    """The production failure needs many cheap jobs; do not write 27,500 files."""
    import os
    from pathlib import Path
    import subprocess
    import sys

    runner = tmp_path / "check_backlog.py"
    runner.write_text("""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from script.robotwin2_prepare_ablation import bounded_results

if __name__ == "__main__":
    consumed = 0
    completed = 0
    def jobs():
        global consumed
        for i in range(27500):
            consumed += 1
            assert consumed - completed <= 20
            yield (i, 0, 0)
    results = []
    with ProcessPoolExecutor(2, mp_context=mp.get_context("spawn")) as pool:
        for job, value in bounded_results(pool, sum, jobs(), 20):
            completed += 1
            results.append(value)
    assert sorted(results) == list(range(27500))
""")
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    subprocess.run(
        [sys.executable, str(runner)],
        env=env,
        check=True,
        timeout=60,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("robot", ["robotwin2", "piper"])
def test_s2_pair_preserves_local_state_and_actions_but_adds_shared_geometry(robot):
    from opendm.dataset.register import CONVERSATION_DATA
    from opendm.data.transforms import _state_desc_value
    from opendm.constants.robot import RobotStateDesc

    joints = np.random.default_rng(41).normal(scale=0.2, size=(12, 14))
    if robot == "robotwin2":
        if not (ROBOTWIN_ROOT / "assets/embodiments/aloha-agilex/config.yml").exists():
            pytest.skip("RoboTwin assets unavailable")
        kin = RoboTwinKinematics()
        build = lambda rep: build_states(joints, kin, rep)
        prefix = "robotwin2_ablation"
    else:
        from script.piper_lerobot_to_jsonl import build_states as piper_states

        build = lambda rep: piper_states(joints, rep)
        prefix = "piper_fold"
    local = build("eef_local")
    paired = build("eef_local_pair")
    shared = build("eef_unified_pair")
    np.testing.assert_array_equal(paired[:, :14], local)
    np.testing.assert_array_equal(paired[:, 14:], shared[:, 14:])
    # Independent local frames cannot be treated as if they shared an origin.
    incorrect = se3.transform_to_pos_rot6d(
        se3.relative_transform(
            se3.pos_rotvec_to_transform(local[:, :6]),
            se3.pos_rotvec_to_transform(local[:, 7:13]),
        )
    )
    assert not np.allclose(paired[:, 14:17], incorrect[:, :3])
    # A first-camera frame is a common left multiplication; it cancels exactly.
    h = sample_episode_frame("first-camera", 19)
    left = h @ se3.pos_rotvec_to_transform(shared[:, :6])
    right = h @ se3.pos_rotvec_to_transform(shared[:, 7:13])
    np.testing.assert_allclose(
        paired[:, 14:],
        se3.transform_to_pos_rot6d(se3.relative_transform(left, right)),
        atol=1e-12,
    )
    info = CONVERSATION_DATA[f"{prefix}_s2_pair"]
    desc = info["state_desc"]
    assert len(desc) == 23
    assert all(
        _state_desc_value(x) == _state_desc_value(RobotStateDesc.AUX) for x in desc[14:]
    )
    assert info["jsonl_dir"].endswith("/eef_local_pair/jsonl/train")

    def encode(states, state_desc):
        return ActionRelativeSE3()(
            {
                "state": states[0],
                "action": states[1:][None],
                "meta_data": {"state_desc": state_desc},
            }
        )["action"]

    action = encode(paired, desc)
    assert action.shape == (1, 11, 20)
    np.testing.assert_array_equal(action, encode(local, desc[:14]))
    poisoned = paired.copy()
    poisoned[:, 14:] += 1000
    np.testing.assert_array_equal(action, encode(poisoned, desc))
    if robot == "piper":
        decoded, grip = decode_to_common_space(
            "piper_fold_s2_pair", paired[0], action[0], desc
        )
        truth, truth_grip = ground_truth_common_space(local[1:], desc[:14], False)
        np.testing.assert_allclose(decoded, truth, atol=2e-6)
        np.testing.assert_allclose(grip, truth_grip, atol=1e-7)


def test_s2_pair_requires_se3_and_has_separate_norm_cache():
    from playground.dm05_robotwin2_ablation import DM05DataConfig as RobotwinConfig
    from playground.dm05_piper import DM05DataConfig as PiperConfig

    for config_type, prefix in [
        (RobotwinConfig, "robotwin2_ablation"),
        (PiperConfig, "piper_fold"),
    ]:
        config = config_type(dataset_name=f"{prefix}_s2_pair", relative_mode="se3")
        assert len(config._dataset_info()["state_desc"]) == 23
        other = config_type(dataset_name=f"{prefix}_s2", relative_mode="se3")
        assert config.norm_stats_path(50) != other.norm_stats_path(50)
        config.relative_mode = "vector"
        with pytest.raises(ValueError, match="requires"):
            config._dataset_info()


def test_s2_pair_can_be_added_without_rewriting_existing_rungs(tmp_path):
    if not (ROBOTWIN_ROOT / "assets/embodiments/aloha-agilex/config.yml").exists():
        pytest.skip("RoboTwin assets unavailable")
    source = tmp_path / "source"
    directory = source / "jsonl/task/clean"
    directory.mkdir(parents=True)
    for i in range(4):
        rows = [
            {
                "state": (np.arange(14) * 0.01 + j * 0.01).tolist(),
                "prompt": "move",
                "images_1": {"url": "unchanged.mp4"},
            }
            for j in range(3)
        ]
        (directory / f"episode{i}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    destination = tmp_path / "derived"
    args = ["--source", str(source), "--out-root", str(destination), "--workers", "1"]
    main(args + ["--rung", "s2", "s3"])
    prior = {p: p.read_bytes() for p in destination.rglob("*.jsonl")}
    split = (destination / "split.json").read_bytes()
    manifest = (destination / "manifest.json").read_bytes()
    main(args + ["--rung", "s2_pair"])
    assert (destination / "split.json").read_bytes() == split
    assert (destination / "manifest.json").read_bytes() == manifest
    assert all(path.read_bytes() == content for path, content in prior.items())
    for local_path in (destination / "eef_local/jsonl").rglob("*.jsonl"):
        relative = local_path.relative_to(destination / "eef_local/jsonl")

        def read(rep):
            return [
                json.loads(line)
                for line in (destination / rep / "jsonl" / relative)
                .read_text()
                .splitlines()
            ]

        local, pair, shared = (
            read("eef_local"),
            read("eef_local_pair"),
            read("eef_unified_pair"),
        )
        for a, b, c in zip(local, pair, shared, strict=True):
            assert b["state"] == a["state"] + c["state"][14:]
            assert b["images_1"] == a["images_1"] and b["prompt"] == a["prompt"]
            assert "action" not in b
