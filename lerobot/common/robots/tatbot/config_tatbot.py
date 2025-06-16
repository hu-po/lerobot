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
    connection_timeout: float = 5.0
    """Timeout when connecting to the robot arms in seconds."""
    joint_tolerance_warning: float = 1e-3
    """Warning tolerance for joint position mismatch."""
    joint_tolerance_error: float = 0.3
    """Error tolerance for joint position error."""

    home_pos_l: list[float] = field(default_factory=lambda: [1.5708 - 0.3] + [0.0] * 6)
    """Radian joint positions of the left arm: folded up, resting on itself, rotated slightly inwards."""
    home_pos_r: list[float] = field(default_factory=lambda: [0.3] + [0.0] * 6)
    """Radian joint positions of the right arm: folded up, resting on itself, rotated slightly inwards."""

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
                serial_number_or_name="218622278376",
            ),
            "wrist": RealSenseCameraConfig(
                fps=30,
                width=640,
                height=480,
                serial_number_or_name="230422273017",
            ),
        }
    )
