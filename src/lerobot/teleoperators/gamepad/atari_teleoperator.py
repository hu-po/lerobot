from dataclasses import dataclass
from typing import Any
import evdev
from evdev import InputDevice, ecodes
import threading
import queue as thread_queue
import logging
from ..teleoperator import Teleoperator
from ..config import TeleoperatorConfig

logger = logging.getLogger(__name__)

ATARI_NAME = "Retro Games LTD  Atari CX Wireless Controller"
RED_BUTTON_CODE = 304  # Detected from debug logs

@TeleoperatorConfig.register_subclass("atari")
@dataclass
class AtariTeleoperatorConfig(TeleoperatorConfig):
    device_name: str = ATARI_NAME
    axis_threshold: float = 0.5
    queue_size: int = 0  # Unbounded queue to avoid silent drops

class AtariTeleoperator(Teleoperator):
    config_class = AtariTeleoperatorConfig
    name = "atari"

    def __init__(self, config: AtariTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self.device = None
        self._queue = thread_queue.Queue(maxsize=config.queue_size)
        self._thread = None
        self._stop_event = threading.Event()
        self._last_axis = {'x': None, 'y': None}
        self._connected = False

    @property
    def action_features(self) -> dict:
        return {"x": float, "y": float, "red_button": bool}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        self.device = self._find_joystick()
        if not self.device:
            raise RuntimeError(f"Joystick '{self.config.device_name}' not found.")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._event_loop, daemon=True)
        self._thread.start()
        self._connected = True

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1)
        self._connected = False

    def _find_joystick(self):
        devices = [InputDevice(path) for path in evdev.list_devices()]
        for d in devices:
            if d.name.strip() == self.config.device_name:
                return d
        return None

    def _event_loop(self):
        try:
            for event in self.device.read_loop():
                if self._stop_event.is_set():
                    break
                if event.type == ecodes.EV_KEY:
                    if event.code == RED_BUTTON_CODE:
                        if event.value == 1:
                            self._put_event({'red_button': True})
                        elif event.value == 0:
                            self._put_event({'red_button': False})
                elif event.type == ecodes.EV_ABS:
                    axis = None
                    if event.code == ecodes.ABS_X:
                        axis = 'x'
                    elif event.code == ecodes.ABS_Y:
                        axis = 'y'
                    if axis:
                        # Use device introspection for robust axis normalization
                        try:
                            absinfo = self.device.absinfo(event.code)
                            if absinfo and absinfo.max != absinfo.min:
                                norm = (event.value - absinfo.min) / (absinfo.max - absinfo.min) * 2 - 1
                            else:
                                # Fallback to hard-coded formula if absinfo is missing or invalid
                                norm = event.value / 127.5 - 1.0
                        except (AttributeError, OSError):
                            # Fallback to hard-coded formula if absinfo fails
                            norm = event.value / 127.5 - 1.0
                        
                        last = self._last_axis[axis]
                        if last is None or abs(norm - last) > self.config.axis_threshold:
                            self._last_axis[axis] = norm
                            self._put_event({axis: norm})
        except Exception as e:
            logger.exception("Joystick loop error: %s", e)

    def _put_event(self, event):
        # Check queue size and warn if it's getting too large
        if self._queue.qsize() > 1000:
            logger.warning(f"Atari teleoperator queue size is {self._queue.qsize()}, consider processing events faster")
        self._queue.put_nowait(event)

    def get_action(self) -> dict[str, Any]:
        # Drain the queue and aggregate all events to capture all inputs
        action = {"x": 0.0, "y": 0.0, "red_button": False}
        while not self._queue.empty():
            evt = self._queue.get_nowait()
            action.update(evt)
        return action

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass 