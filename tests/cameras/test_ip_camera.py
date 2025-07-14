#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE/2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError


class TestIPCameraConfig:
    """Test IP camera configuration."""

    def test_ip_camera_config_basic(self):
        """Test basic IP camera configuration."""
        config = OpenCVCameraConfig(
            ip="192.168.1.100",
            username="admin",
            password="password123",
            rtsp_port=554
        )
        
        assert config.ip == "192.168.1.100"
        assert config.username == "admin"
        assert config.password == "password123"
        assert config.rtsp_port == 554
        assert config.index_or_path == "rtsp://admin:password123@192.168.1.100:554"


class TestIPCamera:
    """Test IP camera functionality."""

    @pytest.fixture
    def mock_cv2(self):
        """Mock OpenCV module."""
        with patch('lerobot.cameras.opencv.camera_opencv.cv2') as mock_cv2:
            # Mock VideoCapture
            mock_videocapture = MagicMock()
            mock_videocapture.isOpened.return_value = True
            mock_videocapture.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FPS: 30.0,
                mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
                mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                mock_cv2.CAP_PROP_FORMAT: 16,
            }.get(prop, 0)
            mock_videocapture.set.return_value = True
            mock_videocapture.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
            mock_videocapture.release.return_value = None
            mock_videocapture.getBackendName.return_value = "FFMPEG"
            
            mock_cv2.VideoCapture.return_value = mock_videocapture
            mock_cv2.setNumThreads.return_value = None
            mock_cv2.cvtColor.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
            mock_cv2.rotate.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
            
            # Mock rotation constants
            mock_cv2.ROTATE_90_CLOCKWISE = 0
            mock_cv2.ROTATE_90_COUNTERCLOCKWISE = 1
            mock_cv2.ROTATE_180 = 2
            mock_cv2.ROTATE_270 = 3
            
            # Mock color conversion constants
            mock_cv2.COLOR_BGR2RGB = 4
            
            yield mock_cv2, mock_videocapture

    @pytest.fixture
    def ip_camera_config(self):
        """Create IP camera configuration for testing."""
        return OpenCVCameraConfig(
            ip="192.168.1.100",
            username="admin",
            password="password123",
            width=1920,
            height=1080,
            fps=30,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION
        )

    @pytest.fixture
    def ip_camera(self, ip_camera_config, mock_cv2):
        """Create IP camera instance with mocked dependencies."""
        mock_cv2_module, mock_videocapture = mock_cv2
        camera = OpenCVCamera(ip_camera_config)
        yield camera, mock_videocapture

    def test_ip_camera_initialization(self, ip_camera):
        """Test IP camera initialization."""
        camera, mock_videocapture = ip_camera
        
        assert camera.config.ip == "192.168.1.100"
        assert camera.config.username == "admin"
        assert camera.config.password == "password123"
        assert camera.config.rtsp_port == 554
        assert camera.index_or_path == "rtsp://admin:password123@192.168.1.100:554"
        assert camera.width == 1920
        assert camera.height == 1080
        assert camera.fps == 30

    def test_ip_camera_connect_and_disconnect(self, ip_camera):
        """Test IP camera connection and disconnection."""
        camera, mock_videocapture = ip_camera
        
        camera.connect(warmup=False)
        assert camera.is_connected
        
        camera.disconnect()
        assert not camera.is_connected
        mock_videocapture.release.assert_called_once()

    def test_ip_camera_connect_failure(self, ip_camera):
        """Test IP camera connection failure."""
        camera, mock_videocapture = ip_camera
        
        # Mock connection failure
        mock_videocapture.isOpened.return_value = False
        
        with pytest.raises(ConnectionError, match="Failed to open"):
            camera.connect(warmup=False)

    def test_ip_camera_read_and_read_before_connect(self, ip_camera):
        """Test IP camera frame reading and reading before connection."""
        camera, mock_videocapture = ip_camera
        
        camera.connect(warmup=False)
        
        # Mock successful frame read
        test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        mock_videocapture.read.return_value = (True, test_frame)
        
        frame = camera.read()
        
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (1080, 1920, 3)
        assert frame.dtype == np.uint8

        camera.disconnect()
        with pytest.raises(DeviceNotConnectedError):
            camera.read()

    def test_ip_camera_str_representation(self, ip_camera):
        """Test IP camera string representation."""
        camera, mock_videocapture = ip_camera
        
        # Should show IP address but not password
        str_repr = str(camera)
        assert "192.168.1.100" in str_repr
        assert "password123" not in str_repr
        assert "OpenCVCamera" in str_repr


if __name__ == "__main__":
    pytest.main([__file__]) 