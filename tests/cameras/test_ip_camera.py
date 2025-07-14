import pytest
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

def test_ip_camera_config_basic():
    config = OpenCVCameraConfig(
        ip="192.168.1.100",
        username="admin",
        password="password123",
        rtsp_port=554,
        width=1920,
        height=1080,
        fps=30,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION
    )
    assert config.ip == "192.168.1.100"
    assert config.username == "admin"
    assert config.password == "password123"
    assert config.rtsp_port == 554
    assert config.index_or_path == "rtsp://admin:password123@192.168.1.100:554"

def test_ip_camera_initialization():
    config = OpenCVCameraConfig(
        ip="192.168.1.100",
        username="admin",
        password="password123",
        width=1920,
        height=1080,
        fps=30
    )
    camera = OpenCVCamera(config)
    assert camera.config.ip == "192.168.1.100"
    assert camera.config.username == "admin"
    assert camera.config.password == "password123"
    assert camera.width == 1920
    assert camera.height == 1080
    assert camera.fps == 30


if __name__ == "__main__":
    pytest.main([__file__]) 