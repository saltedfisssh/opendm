"""RealSense color capture for the robot host; SDK imports are intentionally lazy."""

import base64
import threading
import time

import numpy as np


def load_sdk():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import pyrealsense2 on this host. Install pyrealsense2 in the "
            "deployment environment and its system dependencies (on Ubuntu/Debian: "
            "libusb-1.0-0). Original error: " + str(exc)
        ) from exc
    return rs


def list_devices():
    """Enumerate serials and supported color modes without starting any stream."""
    rs = load_sdk()
    context = rs.context()
    result = []
    for device in context.query_devices():

        def info(key):
            return device.get_info(key) if device.supports(key) else None

        modes = set()
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() == rs.stream.color:
                    video = profile.as_video_stream_profile()
                    modes.add(
                        (
                            video.width(),
                            video.height(),
                            profile.fps(),
                            str(profile.format()),
                        )
                    )
        result.append(
            {
                "serial": info(rs.camera_info.serial_number),
                "name": info(rs.camera_info.name),
                "usb_type": info(rs.camera_info.usb_type_descriptor),
                "color_modes": [
                    dict(width=w, height=h, fps=fps, format=fmt)
                    for w, h, fps, fmt in sorted(modes)
                ],
            }
        )
    return sorted(result, key=lambda item: item["serial"] or "")


class RealSenseCamera:
    """One serial-bound RGB camera, continuously drained into a single latest frame.

    The wire format remains JPEG. Capture uses BGR8 to match OpenCV's encoder;
    the server's JPEG decoder yields RGB, without an extra channel swap.
    Timestamps are host monotonic arrival times, not cross-device hardware time.
    """

    def __init__(self, serial, startup_timeout=10.0, warmup_frames=15):
        if not isinstance(serial, str) or not serial.strip():
            raise ValueError("A RealSense serial number is required")
        if (
            not np.isfinite(startup_timeout)
            or startup_timeout <= 0
            or warmup_frames < 0
        ):
            raise ValueError("Invalid camera startup timeout or warmup frame count")
        import cv2

        rs = load_sdk()
        self.serial = serial
        self.cv2 = cv2
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.ready = threading.Event()
        self.latest = None
        self.error = None
        self.thread = None
        self.started = False
        self.warmup_frames = warmup_frames
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            profile = self.pipeline.start(config)
            self.started = True
            actual = profile.get_device().get_info(rs.camera_info.serial_number)
            if actual != serial:
                raise RuntimeError(
                    f"RealSense serial mismatch: requested {serial}, got {actual}"
                )
            self.thread = threading.Thread(
                target=self._run, name=f"realsense-{serial}", daemon=True
            )
            self.thread.start()
            if not self.ready.wait(startup_timeout):
                raise TimeoutError(
                    f"RealSense {serial}: no color frame after warmup within {startup_timeout}s"
                )
            self.check_fresh(startup_timeout)
        except BaseException:
            self.close()
            raise

    def _run(self):
        last_number = None
        received = 0
        try:
            while not self.closed.is_set():
                frames = self.pipeline.poll_for_frames()
                if not frames:
                    self.closed.wait(0.005)
                    continue
                color = frames.get_color_frame()
                if not color:
                    self.closed.wait(0.005)
                    continue
                number = color.get_frame_number()
                if number == last_number:
                    self.closed.wait(0.005)
                    continue
                last_number = number
                received += 1
                if received <= self.warmup_frames:
                    continue
                # Own the bytes: SDK frame storage may be recycled after this iteration.
                frame = np.asanyarray(color.get_data()).copy()
                if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
                    raise RuntimeError(
                        f"Unexpected color frame shape/dtype: {frame.shape}/{frame.dtype}"
                    )
                with self.lock:
                    self.latest = (time.monotonic(), frame)
                self.ready.set()
        except Exception as exc:
            with self.lock:
                self.error = exc
            self.ready.set()

    def _check_locked(self, max_age):
        if self.closed.is_set():
            raise RuntimeError(f"RealSense {self.serial}: camera is closed")
        if self.error is not None:
            raise RuntimeError(
                f"RealSense {self.serial}: capture failed: {self.error}"
            ) from self.error
        if self.latest is None or time.monotonic() - self.latest[0] > max_age:
            raise RuntimeError(f"RealSense {self.serial}: missing or stale color frame")

    def check_fresh(self, max_age):
        with self.lock:
            self._check_locked(max_age)

    def image(self, max_age):
        with self.lock:
            self._check_locked(max_age)
            frame = self.latest[1].copy()
        ok, jpeg = self.cv2.imencode(".jpg", frame, [self.cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError(f"RealSense {self.serial}: JPEG encoding failed")
        return base64.b64encode(jpeg).decode("ascii")

    def close(self):
        self.closed.set()
        # poll_for_frames is nonblocking, so the worker can exit before stop().
        if self.thread is not None:
            self.thread.join()
        if self.started:
            self.started = False
            self.pipeline.stop()
