from dataclasses import dataclass

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("tatbot")
@dataclass
class TatbotConfig(RobotConfig):

    rs_cameras: dict[str, CameraConfig]
    """Realsense cameras (used at every timestep)."""
    ip_cameras: dict[str, CameraConfig]
    """PoE IP Cameras (used sparingly since slow)"""
    
    ip_address_l: str
    """IP address of the left robot arm."""
    ip_address_r: str
    """IP address of the right robot arm."""
    
    arm_l_config_filepath: str
    """YAML file containing left arm config."""
    arm_r_config_filepath: str
    """YAML file containing right arm config."""
    
    home_pos_l: list[float]
    """Radian joint positions of the left arm: folded up, resting on itself, facing forwards."""
    home_pos_r: list[float]
    """Radian joint positions of the right arm: folded up, resting on itself, facing forwards."""
    
    goal_time: float
    """Default robot travel time."""
    connection_timeout: float 
    """Timeout when connecting to the robot arms in seconds."""

    joint_tolerance_warning: float = 1e-2
    """Warning tolerance for joint position mismatch."""
    joint_tolerance_error: float = 1.0
    """Error tolerance for joint position error."""

    arm_log_level: str = "INFO"
    """Log level for trossen arm drivers. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL."""

    max_workers: int = 16
    """Max number of worker threads in the thread pool. If None, it defaults to min(32, os.cpu_count() * 4)."""