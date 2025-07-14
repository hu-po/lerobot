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

import time
from unittest.mock import MagicMock, patch

import pytest

from lerobot.teleoperators.gamepad.atari_teleoperator import (
    AtariTeleoperator,
    AtariTeleoperatorConfig,
    ATARI_NAME,
    RED_BUTTON_CODE,
)


class TestAtariTeleoperatorConfig:
    """Test the Atari teleoperator configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AtariTeleoperatorConfig()
        assert config.device_name == ATARI_NAME
        assert config.axis_threshold == 0.5
        assert config.queue_size == 0


class TestAtariTeleoperator:
    """Test the Atari teleoperator functionality."""

    @pytest.fixture
    def mock_evdev(self):
        """Mock evdev module and InputDevice."""
        with patch('lerobot.teleoperators.gamepad.atari_teleoperator.evdev') as mock_evdev:
            with patch('lerobot.teleoperators.gamepad.atari_teleoperator.InputDevice') as mock_input_device:
                # Mock device
                mock_device = MagicMock()
                mock_device.name = ATARI_NAME
                mock_input_device.return_value = mock_device
                
                # Mock list_devices
                mock_evdev.list_devices.return_value = ['/dev/input/event0']
                
                yield mock_evdev, mock_input_device, mock_device

    @pytest.fixture
    def teleop(self, mock_evdev):
        """Create a teleoperator instance with mocked dependencies."""
        config = AtariTeleoperatorConfig()
        return AtariTeleoperator(config)

    def test_initialization(self, teleop):
        """Test teleoperator initialization."""
        assert teleop.config.device_name == ATARI_NAME
        assert teleop.config.queue_size == 0
        assert not teleop.is_connected
        assert teleop.action_features == {"x": float, "y": float, "red_button": bool}
        assert teleop.feedback_features == {}

    def test_connect_success_and_disconnect(self, teleop, mock_evdev):
        """Test successful connection and disconnection."""
        mock_evdev, mock_input_device, mock_device = mock_evdev
        
        # Mock the device to be found
        with patch.object(teleop, '_find_joystick', return_value=mock_device):
            teleop.connect()
            
            assert teleop.is_connected
            assert teleop._thread is not None
            assert teleop._thread.is_alive()

            # Then disconnect
            teleop.disconnect()
            assert not teleop.is_connected

    def test_connect_device_not_found(self, teleop, mock_evdev):
        """Test connection failure when device not found."""
        with patch.object(teleop, '_find_joystick', return_value=None):
            with pytest.raises(RuntimeError, match="Joystick.*not found"):
                teleop.connect()

    def test_get_action_default_and_with_events(self, teleop):
        """Test getting action with default values and with queued events."""
        action = teleop.get_action()
        assert action == {"x": 0.0, "y": 0.0, "red_button": False}
        teleop._queue.put({"x": 0.5})
        teleop._queue.put({"red_button": True})
        action = teleop.get_action()
        assert action == {"x": 0.0, "y": 0.0, "red_button": True}

    def test_event_loop_button_and_axis(self, teleop, mock_evdev):
        """Test event loop handling button press and axis movement."""
        mock_evdev, mock_input_device, mock_device = mock_evdev
        
        # Button event
        mock_event_btn = MagicMock()
        mock_event_btn.type = mock_evdev.ecodes.EV_KEY
        mock_event_btn.code = RED_BUTTON_CODE
        mock_event_btn.value = 1
        
        # Axis event
        mock_event_axis = MagicMock()
        mock_event_axis.type = mock_evdev.ecodes.EV_ABS
        mock_event_axis.code = mock_evdev.ecodes.ABS_X
        mock_event_axis.value = 255
        
        # Mock device read_loop
        mock_device.read_loop.return_value = [mock_event_btn, mock_event_axis]
        
        # Connect and start event loop
        with patch.object(teleop, '_find_joystick', return_value=mock_device):
            teleop.connect()
            time.sleep(0.1)  # Give thread time to process
            
            # Check that event was queued
            action = teleop.get_action()
            assert action["red_button"] is True or action["x"] == 1.0
            
            teleop.disconnect()

    def test_interface_methods(self, teleop):
        """Test calibration and configuration methods."""
        # These methods should not raise exceptions
        teleop.calibrate()
        teleop.configure()
        teleop.send_feedback({"test": "data"})
        assert teleop.is_calibrated is True


if __name__ == "__main__":
    pytest.main([__file__]) 