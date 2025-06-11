from dataclasses import dataclass, field

from lerobot.common.cameras import CameraConfig
from lerobot.common.cameras.realsense import RealSenseCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("tatbot")
@dataclass
class TatbotConfig(RobotConfig):
    goal_time_fast: float = 0.5
    """Robot travel time when executing fast actions, usually small movements."""
    goal_time_slow: float = 3.0
    """Robot travel time when moving slowly, usually larger movements."""

    ip_address_l: str = "192.168.1.3"
    """IP address of the left robot arm."""
    ip_address_r: str = "192.168.1.2"
    """IP address of the right robot arm."""

    # cameras
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "head": RealSenseCameraConfig(
                fps=30,
                width=640,
                height=480,
                serial_number_or_name="218622278376", # realsense_b
            ),
            "wrist": RealSenseCameraConfig(
                fps=30,
                width=640,
                height=480,
                serial_number_or_name="230422273017", # realsense_a
            ),
        }
    )
