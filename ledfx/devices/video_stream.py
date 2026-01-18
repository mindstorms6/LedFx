import logging
import queue
import subprocess
import threading
import time
from typing import Optional

import numpy as np
import voluptuous as vol

from ledfx.devices import Device
from ledfx.events import DevicesUpdatedEvent
from ledfx.utils import BaseRegistry

_LOGGER = logging.getLogger(__name__)


# This wrapper is required to prevent config_update lifecycle breakage
# You cannot inherit from Device directly
@BaseRegistry.no_registration
class DeviceWrapper(Device):
    pass


class VideoStreamDevice(DeviceWrapper):
    """Video stream device using ffmpeg (raw RGB frames -> RTSP/UDP/etc)."""

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Required(
                "pixel_count",
                description="Number of individual pixels (rows * cols)",
                default=1,
            ): vol.All(int, vol.Range(min=1)),
            vol.Required(
                "rows",
                description="Number of rows in the 2D matrix",
                default=2,
            ): vol.All(int, vol.Range(min=2)),
            vol.Required(
                "cols",
                description="Number of columns in the 2D matrix",
                default=2,
            ): vol.All(int, vol.Range(min=2)),
            vol.Required(
                "output_url",
                description="Stream destination (rtsp://, udp://, rtmp://, etc)",
                default="udp://127.0.0.1:1234?pkt_size=1316",
            ): str,
            vol.Optional(
                "output_format",
                description="ffmpeg output format (e.g. rtsp, mpegts)",
                default="mpegts",
            ): str,
            vol.Optional(
                "rtsp_transport",
                description="RTSP transport (udp, tcp, udp_multicast, http, https)",
                default="tcp",
            ): str,
            vol.Optional(
                "rtsp_listen",
                description="Enable RTSP server mode (ffmpeg -rtsp_flags listen)",
                default=True,
            ): bool,
            vol.Optional(
                "stream_fps",
                description="Stream FPS (encoder input rate)",
                default=30,
            ): vol.All(int, vol.Range(min=1, max=240)),
            vol.Optional(
                "video_encoder",
                description="FFmpeg video encoder (e.g. libx264, h264_videotoolbox)",
                default="libx264",
            ): str,
            vol.Optional(
                "encoder_preset",
                description="FFmpeg encoder preset (if supported by encoder)",
                default="ultrafast",
            ): str,
            vol.Optional(
                "output_pix_fmt",
                description="Output pixel format (e.g. yuv420p, yuv422p)",
                default="yuv420p",
            ): str,
            vol.Optional(
                "crf",
                description="CRF quality target (libx264 only)",
                default=23,
            ): vol.All(int, vol.Range(min=0, max=51)),
            vol.Optional(
                "video_bitrate",
                description="Target video bitrate (e.g. 4M, 800k). Overrides CRF for some encoders.",
                default="",
            ): str,
            vol.Optional(
                "low_latency",
                description="Apply extra ffmpeg low-latency flags",
                default=True,
            ): bool,
            vol.Optional(
                "ffmpeg_path",
                description="Path to ffmpeg binary",
                default="ffmpeg",
            ): str,
        }
    )

    def __init__(self, ledfx, config):
        super().__init__(ledfx, config)
        self._device_type = "Video Stream"
        self._rows = self._config["rows"]
        self._cols = self._config["cols"]
        self._process: Optional[subprocess.Popen] = None
        self._frame_queue: Optional[queue.Queue] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stream_lock = threading.Lock()
        self._drop_warning_logged = False
        self._next_frame_time = 0.0

    def config_updated(self, config):
        self._rows = config["rows"]
        self._cols = config["cols"]
        self._next_frame_time = 0.0
        expected_pixels = self._rows * self._cols
        if config["pixel_count"] != expected_pixels:
            _LOGGER.warning(
                "Video Stream device %s pixel_count (%s) != rows*cols (%s).",
                self.name,
                config["pixel_count"],
                expected_pixels,
            )
        if self._active:
            self.deactivate()
            self.activate()

    def _build_ffmpeg_cmd(self):
        size = f"{self._cols}x{self._rows}"
        fps = str(self._config["stream_fps"])
        output_url = self._config["output_url"]
        cmd = [
            self._config["ffmpeg_path"],
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            size,
            "-r",
            fps,
            "-i",
            "-",
            "-an",
            "-c:v",
            self._config.get("video_encoder", "libx264"),
            "-g",
            fps,
            "-keyint_min",
            fps,
        ]
        encoder_preset = self._config.get("encoder_preset")
        if encoder_preset:
            cmd.extend(["-preset", encoder_preset])
        output_pix_fmt = self._config.get("output_pix_fmt")
        if output_pix_fmt:
            cmd.extend(["-pix_fmt", output_pix_fmt])
        video_bitrate = self._config.get("video_bitrate")
        if video_bitrate:
            cmd.extend(["-b:v", str(video_bitrate)])
        elif self._config.get("video_encoder", "libx264") == "libx264":
            cmd.extend(["-crf", str(self._config.get("crf", 23))])
        if self._config.get("video_encoder", "libx264") == "libx264":
            cmd.extend(["-tune", "zerolatency"])
        if self._config.get("low_latency", True):
            cmd.extend(
                [
                    "-bf",
                    "0",
                    "-sc_threshold",
                    "0",
                    "-muxdelay",
                    "0",
                    "-muxpreload",
                    "0",
                    "-flush_packets",
                    "1",
                ]
            )
        output_format = self._config.get("output_format")
        if output_format:
            cmd.extend(["-f", output_format])
        if output_format == "rtsp":
            rtsp_transport = self._config.get("rtsp_transport", "tcp")
            if self._config.get("rtsp_listen", True) and rtsp_transport == "udp":
                _LOGGER.warning(
                    "Video Stream device %s rtsp_listen is enabled but rtsp_transport is udp; "
                    "some ffmpeg builds will still attempt client mode. Consider tcp.",
                    self.name,
                )
            cmd.extend(["-rtsp_transport", rtsp_transport])
            if self._config.get("rtsp_listen", True):
                cmd.extend(["-rtsp_flags", "listen"])
                if "listen=1" not in output_url:
                    separator = "&" if "?" in output_url else "?"
                    output_url = f"{output_url}{separator}listen=1"
                output_url = output_url.replace(
                    "rtsp://localhost", "rtsp://0.0.0.0"
                ).replace("rtsp://127.0.0.1", "rtsp://0.0.0.0")
        cmd.append(output_url)
        return cmd

    def _stream_worker(self):
        assert self._frame_queue is not None
        assert self._process is not None
        try:
            while self._active and self._process and self._process.stdin:
                try:
                    frame = self._frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if frame is None:
                    break
                try:
                    self._process.stdin.write(frame)
                except BrokenPipeError as exc:
                    _LOGGER.warning(
                        "Video Stream device %s ffmpeg pipe closed: %s",
                        self.name,
                        exc,
                    )
                    self._online = False
                    self._ledfx.events.fire_event(
                        DevicesUpdatedEvent(self.id)
                    )
                    break
        finally:
            if self._process and self._process.stdin:
                try:
                    self._process.stdin.close()
                except Exception:
                    pass

    def _stderr_worker(self):
        assert self._process is not None
        stderr = self._process.stderr
        if stderr is None:
            return
        try:
            for line in iter(stderr.readline, b""):
                message = line.decode("utf-8", errors="replace").strip()
                if message:
                    _LOGGER.warning(
                        "Video Stream device %s ffmpeg: %s",
                        self.name,
                        message,
                    )
        except Exception as exc:
            _LOGGER.debug(
                "Video Stream device %s stderr reader stopped: %s",
                self.name,
                exc,
            )

    def activate(self):
        with self._stream_lock:
            if self._process is not None:
                return
            try:
                cmd = self._build_ffmpeg_cmd()
                _LOGGER.info(
                    "Starting Video Stream device %s: %s", self.name, cmd
                )
                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
                self._frame_queue = queue.Queue(maxsize=2)
                # Activate before starting threads to avoid race exiting worker
                super().activate()
                self._stream_thread = threading.Thread(
                    name=f"VideoStream: {self.id}",
                    target=self._stream_worker,
                    daemon=True,
                )
                self._stream_thread.start()
                self._stderr_thread = threading.Thread(
                    name=f"VideoStream stderr: {self.id}",
                    target=self._stderr_worker,
                    daemon=True,
                )
                self._stderr_thread.start()
                self._online = True
                _LOGGER.info(
                    "Started Video Stream device %s: %s pid %s", self.name, cmd, self._process.pid
                )
            except Exception as exc:
                _LOGGER.warning(
                    "Failed to start Video Stream device %s: %s",
                    self.name,
                    exc,
                )
                self._process = None
                self._frame_queue = None
                self._stream_thread = None
                self._stderr_thread = None
                self._online = False
                self._ledfx.events.fire_event(DevicesUpdatedEvent(self.id))
                return

    def deactivate(self):
        with self._stream_lock:
            self._active = False
            if self._frame_queue is not None:
                try:
                    self._frame_queue.put_nowait(None)
                except queue.Full:
                    pass
            if self._stream_thread is not None:
                self._stream_thread.join(timeout=2)
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=2)
            if self._process is not None:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=2)
                except Exception:
                    pass
            self._process = None
            self._frame_queue = None
            self._stream_thread = None
            self._stderr_thread = None
        super().deactivate()

    def flush(self, data):
        if not self._active or self._process is None:
            return
        if data is None:
            return

        expected_pixels = self._rows * self._cols
        if data.shape[0] != expected_pixels:
            _LOGGER.warning(
                "Video Stream device %s frame size mismatch (%s != %s).",
                self.name,
                data.shape[0],
                expected_pixels,
            )
            return

        frame = np.clip(data, 0, 255).astype(np.uint8, copy=False)
        try:
            frame = frame.reshape((self._rows, self._cols, 3))
        except ValueError:
            _LOGGER.warning(
                "Video Stream device %s failed to reshape frame.", self.name
            )
            return

        payload = frame.tobytes()

        if self._frame_queue is None:
            return

        stream_fps = max(1, int(self._config.get("stream_fps", 30)))
        now = time.monotonic()
        if now < self._next_frame_time:
            if not self._drop_warning_logged:
                _LOGGER.debug(
                    "Video Stream device %s dropping frames (rate limit).",
                    self.name,
                )
                self._drop_warning_logged = True
            return
        self._next_frame_time = now + (1.0 / stream_fps)

        if self._process and self._process.poll() is not None:
            _LOGGER.warning(
                "Video Stream device %s ffmpeg exited with code %s.",
                self.name,
                self._process.returncode,
            )
            self._online = False
            self._ledfx.events.fire_event(DevicesUpdatedEvent(self.id))
            return

        try:
            self._frame_queue.put_nowait(payload)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(payload)
                if not self._drop_warning_logged:
                    _LOGGER.debug(
                        "Video Stream device %s dropping frames (encoder lag).",
                        self.name,
                    )
                    self._drop_warning_logged = True
            except queue.Empty:
                pass
