import pytest
from lerobot.robots.tatbot.tatbot import Tatbot
from lerobot.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

def make_basic_tatbot_config():
    cameras = {"main": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)}
    cond_cameras = {"cond": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30)}
    return TatbotConfig(
        cameras=cameras,
        cond_cameras=cond_cameras,
        ip_address_l="192.168.1.100",
        ip_address_r="192.168.1.101",
        arm_l_config_filepath="~/config/arm_l.yaml",
        arm_r_config_filepath="~/config/arm_r.yaml",
        home_pos_l=[0.0]*7,
        home_pos_r=[0.0]*7,
        goal_time_fast=0.5,
        goal_time_slow=2.0,
        connection_timeout=5.0
    )

def test_joints_generation():
    config = make_basic_tatbot_config()
    tatbot = Tatbot(config)
    expected_joints = [
        "left.joint_0", "left.joint_1", "left.joint_2", "left.joint_3", 
        "left.joint_4", "left.joint_5", "left.gripper",
        "right.joint_0", "right.joint_1", "right.joint_2", "right.joint_3", 
        "right.joint_4", "right.joint_5", "right.gripper"
    ]
    assert tatbot.joints == expected_joints
    assert len(tatbot.joints) == 14

def test_urdf_joints_to_action():
    config = make_basic_tatbot_config()
    tatbot = Tatbot(config)
    urdf_joints = [float(i) for i in range(15)]
    action = tatbot._urdf_joints_to_action(urdf_joints)
    for i, joint in enumerate(tatbot.joints):
        if i < 7:
            expected_value = float(i)
        else:
            expected_value = float(i + 1)
        assert action[f"{joint}.pos"] == expected_value


if __name__ == "__main__":
    pytest.main([__file__]) 