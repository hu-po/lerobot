import pytest

from lerobot.teleoperators.gamepad.atari_teleoperator import (
    AtariTeleoperator,
    AtariTeleoperatorConfig,
    ATARI_NAME,
)


def test_default_config():
    config = AtariTeleoperatorConfig()
    assert config.device_name == ATARI_NAME
    assert config.axis_threshold == 0.5
    assert config.queue_size == 0

def test_initialization():
    config = AtariTeleoperatorConfig()
    teleop = AtariTeleoperator(config)
    assert teleop.config.device_name == ATARI_NAME
    assert teleop.config.queue_size == 0
    assert not teleop.is_connected
    assert teleop.action_features == {"x": float, "y": float, "red_button": bool}
    assert teleop.feedback_features == {}

def test_get_action_default():
    config = AtariTeleoperatorConfig()
    teleop = AtariTeleoperator(config)
    action = teleop.get_action()
    assert action == {"x": 0.0, "y": 0.0, "red_button": False}


if __name__ == "__main__":
    pytest.main([__file__]) 