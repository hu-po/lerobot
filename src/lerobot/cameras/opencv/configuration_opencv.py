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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os

from ..configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("opencv")
@dataclass
class OpenCVCameraConfig(CameraConfig):
    """Configuration class for OpenCV-based camera devices or video files.

    This class provides configuration options for cameras accessed through OpenCV,
    supporting both physical camera devices and video files. It includes settings
    for resolution, frame rate, color mode, and image rotation.

    Example configurations:
    ```python
    # Basic configurations
    OpenCVCameraConfig(0, 30, 1280, 720)   # 1280x720 @ 30FPS
    OpenCVCameraConfig(/dev/video4, 60, 640, 480)   # 640x480 @ 60FPS

    # Advanced configurations
    OpenCVCameraConfig(128422271347, 30, 640, 480, rotation=Cv2Rotation.ROTATE_90)     # With 90° rotation
    ```

    Attributes:
        index_or_path: Either an integer representing the camera device index,
                      or a Path object pointing to a video file.
                      For IP cameras, this is ignored.
        fps: Requested frames per second for the color stream.
        width: Requested frame width in pixels for the color stream.
        height: Requested frame height in pixels for the color stream.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds)
        ip: IP address of the camera. If provided, IP camera mode is enabled.
        username: Username for RTSP stream authentication.
        password: Password for RTSP stream authentication. Can be an env var like `${VAR}`.
        rtsp_port: Port for the RTSP stream.

    Note:
        - Only 3-channel color output (RGB/BGR) is currently supported.
    """

    index_or_path: Optional[int | Path] = None
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    # IP camera settings
    ip: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    rtsp_port: int = 554

    def __post_init__(self):
        if self.ip:
            self.index_or_path = f"rtsp://{self.username}:{self.password}@{self.ip}:{self.rtsp_port}"

        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` is expected to be in {(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, but {self.rotation} is provided."
            )
