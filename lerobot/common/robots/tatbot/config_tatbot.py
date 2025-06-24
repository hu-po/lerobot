from dataclasses import dataclass, field
import os

from lerobot.common.cameras import CameraConfig
from lerobot.common.cameras.realsense import RealSenseCameraConfig
from lerobot.common.cameras.opencv import OpenCVCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("tatbot")
@dataclass
class TatbotConfig(RobotConfig):
    goal_time_fast: float = 0.2
    """Robot travel time when executing fast actions, usually small movements."""
    goal_time_slow: float = 2.8
    """Robot travel time when moving slowly, usually larger movements."""
    connection_timeout: float = 5.0
    """Timeout when connecting to the robot arms in seconds."""
    joint_tolerance_warning: float = 1e-3
    """Warning tolerance for joint position mismatch."""
    joint_tolerance_error: float = 0.3
    """Error tolerance for joint position error."""

    home_pos_l: list[float] = field(default_factory=lambda: [0.0] * 7)
    """Radian joint positions of the left arm: folded up, resting on itself, facing forwards."""
    home_pos_r: list[float] = field(default_factory=lambda: [0.0] * 7)
    """Radian joint positions of the right arm: folded up, resting on itself, facing forwards."""

    ip_address_l: str = "192.168.1.3"
    """IP address of the left robot arm."""
    arm_l_config_filepath: str = os.path.expanduser("~/tatbot/config/trossen_arm_l.yaml")
    """YAML file containing left arm config."""

    ip_address_r: str = "192.168.1.2"
    """IP address of the right robot arm."""
    arm_r_config_filepath: str = os.path.expanduser("~/tatbot/config/trossen_arm_r.yaml")
    """YAML file containing right arm config."""

    # cameras
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "realsense1": RealSenseCameraConfig(
                fps=30,
                width=640,
                height=480,
                serial_number_or_name="218622278376",
            ),
            "realsense2": RealSenseCameraConfig(
                fps=30,
                width=640,
                height=480,
                serial_number_or_name="230422273017",
            ),
        }
    )

# in bot-only mode, do not use any cameras
TatbotBotOnlyConfig = TatbotConfig(
    cameras={},
)

# in scan mode, use all the cameras at max resolution
TatbotScanConfig = TatbotConfig(
    cameras={
            "realsense1": RealSenseCameraConfig(
                fps=5,
                width=1280,
                height=720,
                serial_number_or_name="218622278376",
                warmup_s=2,
            ),
            "realsense2": RealSenseCameraConfig(
                fps=5,
                width=1280,
                height=720,
                serial_number_or_name="230422273017",
                warmup_s=2,
            ),
            "camera1": OpenCVCameraConfig(
                ip="192.168.1.91",
                username="admin",
                password="${CAMERA_1_PASSWORD}",
                rtsp_port=554,
                stream_path="/cam/realmonitor?channel=1&subtype=0",
                width=2960,
                height=1668,
                fps=5,
            ),
            "camera2": OpenCVCameraConfig(
                ip="192.168.1.92",
                username="admin",
                password="${CAMERA_2_PASSWORD}",
                rtsp_port=554,
                stream_path="/cam/realmonitor?channel=1&subtype=0",
                width=2960,
                height=1668,
                fps=5,
            ),
            "camera3": OpenCVCameraConfig(
                ip="192.168.1.93",
                username="admin",
                password="${CAMERA_3_PASSWORD}",
                rtsp_port=554,
                stream_path="/cam/realmonitor?channel=1&subtype=0",
                width=2960,
                height=1668,
                fps=5,
            ),
            "camera4": OpenCVCameraConfig(
                ip="192.168.1.94",
                username="admin",
                password="${CAMERA_4_PASSWORD}",
                rtsp_port=554,
                stream_path="/cam/realmonitor?channel=1&subtype=0",
                width=2960,
                height=1668,
                fps=5,
            ),
            "camera5": OpenCVCameraConfig(
                ip="192.168.1.95",
                username="admin",
                password="${CAMERA_5_PASSWORD}",
                rtsp_port=554,
                stream_path="/cam/realmonitor?channel=1&subtype=0",
                width=2960,
                height=1668,
                fps=5,
            ),
        }
)