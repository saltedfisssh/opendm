"""Deterministic, invertible S4 coordinate randomization (+z unchanged)."""

import hashlib
import json
import numpy as np
from opendm.data import se3


def sample_episode_frame(episode_id: str, seed: int = 0, xy_range: float = 0.5):
    if not np.isfinite(xy_range) or xy_range < 0:
        raise ValueError("xy_range must be finite and nonnegative")
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:16], "little"))
    yaw = rng.uniform(-np.pi, np.pi)
    return se3.make_transform(
        np.r_[rng.uniform(-xy_range, xy_range, 2), 0.0],
        se3.rotvec_to_mat(np.array([0.0, 0.0, yaw])),
    )


def transform_eef_states(states, transform):
    """Left-multiply both EEF poses; preserve grippers and optional body AUX."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] not in (14, 23):
        raise ValueError("Expected episode states of shape (T, 14 or 23)")
    result = states.copy()
    for start in (0, 7):
        poses = transform @ se3.pos_rotvec_to_transform(states[:, start : start + 6])
        result[:, start : start + 3] = poses[:, :3, 3]
        result[:, start + 3 : start + 6] = se3.unwrap_rotvec_sequence(
            se3.mat_to_rotvec(poses[:, :3, :3])
        )
    return result


def undo_episode_frame(poses, transform):
    """Map decoded (..., 4, 4) poses in G back to W before robot-specific decoding."""
    return se3.transform_inverse(np.asarray(transform)) @ poses


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def write_episode(path, rows, states):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as out:
        for row, state in zip(rows, states, strict=True):
            record = dict(row)
            record["state"] = state.tolist()
            out.write(json.dumps(record, separators=(",", ":")) + "\n")
    tmp.replace(path)
