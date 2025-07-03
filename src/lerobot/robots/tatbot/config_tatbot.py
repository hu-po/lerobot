from dataclasses import dataclass

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("tatbot")
@dataclass
class TatbotConfig(RobotConfig):

    cameras: dict[str, CameraConfig]
    """Cameras are used at every timestep."""
    cond_cameras: dict[str, CameraConfig]
    """Conditioning cameras are used at the start and end of an episode"""
    
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
    
    goal_time_fast: float
    """Robot travel time when executing fast actions, usually small movements."""
    goal_time_slow: float
    """Robot travel time when moving slowly, usually larger movements."""
    connection_timeout: float 
    """Timeout when connecting to the robot arms in seconds."""

    joint_tolerance_warning: float = 1e-2
    """Warning tolerance for joint position mismatch."""
    joint_tolerance_error: float = 1.0
    """Error tolerance for joint position error."""
