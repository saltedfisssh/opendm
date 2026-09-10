"""Exercise SDK behavior with fake devices; never discover or open real hardware."""

import base64
from queue import Empty, Queue
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from opendm.deploy import realsense


@pytest.fixture
def sdk(monkeypatch):
    queue = Queue()
    calls = []

    class Config:
        def enable_device(self, serial):
            calls.append(("serial", serial))

        def enable_stream(self, *args):
            calls.append(("stream", *args))

    class Pipeline:
        def start(self, config):
            calls.append(("start",))
            return SimpleNamespace(
                get_device=lambda: SimpleNamespace(get_info=lambda _: "012345")
            )

        def poll_for_frames(self):
            try:
                value = queue.get_nowait()
            except Empty:
                return None
            if isinstance(value, Exception):
                raise value
            return value

        def stop(self):
            calls.append(("stop",))

    rs = SimpleNamespace(
        pipeline=Pipeline,
        config=Config,
        stream=SimpleNamespace(color="color"),
        format=SimpleNamespace(bgr8="bgr8"),
        camera_info=SimpleNamespace(
            serial_number="serial", name="name", usb_type_descriptor="usb"
        ),
    )
    monkeypatch.setattr(realsense, "load_sdk", lambda: rs)
    return SimpleNamespace(rs=rs, queue=queue, calls=calls)


def frame(number, data=None):
    if data is None:
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        data[:, :, 2] = 255  # BGR red; verify JPEG isn't accidentally blue.
    color = SimpleNamespace(get_frame_number=lambda: number, get_data=lambda: data)
    return SimpleNamespace(get_color_frame=lambda: color)


def test_serial_stream_warmup_color_and_owned_frame(sdk):
    import cv2

    for i in range(1, 16):
        sdk.queue.put(frame(i))
    data = np.zeros((480, 640, 3), dtype=np.uint8)
    data[:, :, 2] = 255
    sdk.queue.put(frame(16, data))
    camera = realsense.RealSenseCamera("012345", startup_timeout=1)
    try:
        assert sdk.calls[:3] == [
            ("serial", "012345"),
            ("stream", "color", 640, 480, "bgr8", 30),
            ("start",),
        ]
        data[:] = 0  # SDK buffer reuse must not mutate cached image.
        encoded = camera.image(1)
        decoded = cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded.shape == (480, 640, 3)
        assert decoded[100, 100, 2] > 250
        assert decoded[100, 100, 0] < 5
    finally:
        camera.close()
    camera.close()
    assert sdk.calls.count(("stop",)) == 1
    assert not camera.thread.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        camera.image(1)


def test_warmup_requires_enough_frames_and_releases_pipeline(sdk):
    sdk.queue.put(frame(1))
    with pytest.raises(TimeoutError, match="warmup"):
        realsense.RealSenseCamera("012345", startup_timeout=0.03, warmup_frames=1)
    assert ("stop",) in sdk.calls


def test_repeated_frame_number_does_not_refresh_image(sdk):
    sdk.queue.put(frame(7))
    camera = realsense.RealSenseCamera("012345", startup_timeout=1, warmup_frames=0)
    try:
        with camera.lock:
            old = camera.latest[0] - 2
            camera.latest = (old, camera.latest[1])
        sdk.queue.put(frame(7))
        threading.Event().wait(0.02)
        with pytest.raises(RuntimeError, match="stale"):
            camera.check_fresh(0.5)
        assert camera.latest[0] == old
    finally:
        camera.close()


def test_capture_error_not_hidden_by_cached_frame(sdk):
    sdk.queue.put(frame(1))
    camera = realsense.RealSenseCamera("012345", startup_timeout=1, warmup_frames=0)
    try:
        sdk.queue.put(RuntimeError("USB disconnected"))
        camera.thread.join(timeout=1)
        with pytest.raises(RuntimeError, match="USB disconnected"):
            camera.image(10)
    finally:
        camera.close()


def test_bad_color_shape_cleans_up(sdk):
    sdk.queue.put(frame(1, np.zeros((10, 10, 3), dtype=np.uint8)))
    with pytest.raises(RuntimeError, match="shape/dtype"):
        realsense.RealSenseCamera("012345", startup_timeout=1, warmup_frames=0)
    assert ("stop",) in sdk.calls


def test_serial_mismatch_cleans_up(sdk):
    with pytest.raises(RuntimeError, match="serial mismatch"):
        realsense.RealSenseCamera("wrong-serial", startup_timeout=1)
    assert ("stop",) in sdk.calls


def test_enumeration_reports_serial_and_modes_without_starting(sdk):
    def profile(kind):
        return SimpleNamespace(
            stream_type=lambda: kind,
            fps=lambda: 30,
            format=lambda: "bgr8",
            as_video_stream_profile=lambda: SimpleNamespace(
                width=lambda: 640, height=lambda: 480
            ),
        )

    device = SimpleNamespace(
        supports=lambda key: key != "usb",
        get_info=lambda key: {"serial": "012345", "name": "D435"}[key],
        query_sensors=lambda: [
            SimpleNamespace(
                get_stream_profiles=lambda: [profile("color"), profile("depth")]
            )
        ],
    )
    sdk.rs.context = lambda: SimpleNamespace(query_devices=lambda: [device])
    assert realsense.list_devices() == [
        {
            "serial": "012345",
            "name": "D435",
            "usb_type": None,
            "color_modes": [{"width": 640, "height": 480, "fps": 30, "format": "bgr8"}],
        }
    ]
    assert sdk.calls == []
