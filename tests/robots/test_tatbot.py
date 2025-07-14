#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.tatbot import Tatbot, TatbotConfig


class TestTatbotConfig:
    """Test the Tatbot configuration."""

    def test_config_initialization(self):
        """Test TatbotConfig initialization."""
        cameras = {"main": OpenCVCameraConfig(index_or_path=0)}
        cond_cameras = {"cond": OpenCVCameraConfig(index_or_path=1)}
        
        config = TatbotConfig(
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
        
        assert config.ip_address_l == "192.168.1.100"
        assert config.ip_address_r == "192.168.1.101"
        assert config.goal_time_fast == 0.5
        assert config.goal_time_slow == 2.0
        assert config.connection_timeout == 5.0


class TestTatbot:
    """Test the Tatbot robot functionality."""

    @pytest.fixture
    def mock_trossen_arm(self):
        """Mock trossen_arm module."""
        with patch('lerobot.robots.tatbot.tatbot.trossen_arm') as mock_arm:
            # Mock arm driver
            mock_driver = MagicMock()
            mock_driver.get_all_positions.return_value = [0.1]*7
            mock_driver.set_all_positions.return_value = None
            mock_driver.get_error_information.return_value = "No error"
            mock_driver.set_all_modes.return_value = None
            
            # Mock TrossenArmDriver constructor
            mock_arm.TrossenArmDriver.return_value = mock_driver
            
            # Mock enums
            mock_arm.Model.wxai_v0 = "wxai_v0"
            mock_arm.StandardEndEffector.wxai_v0_base = "wxai_v0_base"
            mock_arm.Mode.position = "position"
            mock_arm.Mode.idle = "idle"
            mock_arm.VectorDouble = lambda x: x
            
            yield mock_arm, mock_driver

    @pytest.fixture
    def mock_cameras(self):
        """Mock camera creation."""
        with patch('lerobot.robots.tatbot.tatbot.make_cameras_from_configs') as mock_make_cameras:
            # Create mock cameras
            mock_camera = MagicMock()
            mock_camera.is_connected = True
            mock_camera.async_read.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_camera.height = 480
            mock_camera.width = 640
            mock_camera.connect.return_value = None
            mock_camera.disconnect.return_value = None
            
            mock_make_cameras.return_value = {"main": mock_camera}
            
            yield mock_make_cameras, {"main": mock_camera}

    @pytest.fixture
    def tatbot_config(self):
        """Create a TatbotConfig for testing."""
        cameras = {"main": OpenCVCameraConfig(index_or_path=0)}
        cond_cameras = {"cond": OpenCVCameraConfig(index_or_path=1)}
        
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

    @pytest.fixture
    def tatbot(self, tatbot_config, mock_trossen_arm, mock_cameras):
        """Create a Tatbot instance with mocked dependencies."""
        mock_arm, mock_driver = mock_trossen_arm
        mock_make_cameras, mock_camera_dict = mock_cameras
        
        # Mock os.path.expanduser
        with patch('lerobot.robots.tatbot.tatbot.os.path.expanduser', return_value="/home/user/config/arm.yaml"):
            robot = Tatbot(tatbot_config)
            yield robot

    def test_joints_generation(self, tatbot):
        """Test Tatbot initialization."""
        # Test programmatic joints generation (T-1 TODO)
        expected_joints = [
            "left.joint_0", "left.joint_1", "left.joint_2", "left.joint_3", 
            "left.joint_4", "left.joint_5", "left.gripper",
            "right.joint_0", "right.joint_1", "right.joint_2", "right.joint_3", 
            "right.joint_4", "right.joint_5", "right.gripper"
        ]
        assert tatbot.joints == expected_joints
        assert len(tatbot.joints) == 14

    def test_connect_and_disconnect(self, tatbot, mock_trossen_arm, mock_cameras):
        """Test connect and disconnect methods."""
        mock_arm, mock_driver = mock_trossen_arm
        mock_make_cameras, mock_camera_dict = mock_cameras
        
        # Mock cameras
        tatbot.cameras = mock_camera_dict
        tatbot.cond_cameras = mock_camera_dict
        
        # Test connection
        tatbot.connect()
        assert tatbot.is_connected
        
        # Test disconnection
        tatbot.disconnect()
        assert not tatbot.is_connected

    def test_get_observation(self, tatbot, mock_trossen_arm, mock_cameras):
        """Test get_observation method with parallel camera reading (T-6 TODO)."""
        mock_arm, mock_driver = mock_trossen_arm
        mock_make_cameras, mock_camera_dict = mock_cameras
        
        # Mock the arms
        tatbot.arm_l = mock_driver
        tatbot.arm_r = mock_driver
        
        # Mock cameras
        tatbot.cameras = mock_camera_dict
        
        # Test observation
        obs = tatbot.get_observation()
        
        # Check motor positions (T-5 TODO - built in one go)
        for joint in tatbot.joints:
            assert f"{joint}.pos" in obs
            assert isinstance(obs[f"{joint}.pos"], float)
        
        # Check camera data
        assert "main" in obs
        assert isinstance(obs["main"], np.ndarray)
        assert obs["main"].shape == (480, 640, 3)

    def test_send_action(self, tatbot, mock_trossen_arm):
        """Test send_action with parallel arm commands (T-7 TODO)."""
        mock_arm, mock_driver = mock_trossen_arm
        
        # Mock the arms
        tatbot.arm_l = mock_driver
        tatbot.arm_r = mock_driver
        
        # Create action
        action = {}
        for joint in tatbot.joints:
            action[f"{joint}.pos"] = 0.5
        
        # Test action sending
        result = tatbot.send_action(action, block="both")
        
        # Verify both arms were commanded
        assert mock_driver.set_all_positions.call_count == 2
        assert result == action

    def test_urdf_joints_to_action(self, tatbot):
        """Test _urdf_joints_to_action method (T-8 TODO)."""
        # Create test URDF joints (15 elements, skipping index 7)
        urdf_joints = [float(i) for i in range(15)]
        
        action = tatbot._urdf_joints_to_action(urdf_joints)
        
        # Check that all joints are mapped correctly
        for i, joint in enumerate(tatbot.joints):
            if i < 7:
                expected_value = float(i)
            else:
                expected_value = float(i + 1)  # Skip index 7
            assert action[f"{joint}.pos"] == expected_value

    def test_get_conditioning(self, tatbot, mock_trossen_arm, mock_cameras):
        """Test get_conditioning method."""
        mock_arm, mock_driver = mock_trossen_arm
        mock_make_cameras, mock_camera_dict = mock_cameras
        
        # Mock cameras
        tatbot.cameras = mock_camera_dict
        tatbot.cond_cameras = mock_camera_dict
        
        # Test conditioning
        conditioning = tatbot.get_conditioning()
        
        # Check that both normal and conditioning cameras are included
        assert "main" in conditioning
        assert isinstance(conditioning["main"], np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__]) 