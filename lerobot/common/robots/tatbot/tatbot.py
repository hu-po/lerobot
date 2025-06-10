import logging
import time
from functools import cached_property
from typing import Any

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

import trossen_arm

from ..robot import Robot
from .config_tatbot import TatbotConfig

logger = logging.getLogger(__name__)


class Tatbot(Robot):
    """
    [Tatbot](https://github.com/hu-po/tatbot)
    """

    config_class = TatbotConfig
    name = "tatbot"

    def __init__(self, config: TatbotConfig):
        super().__init__(config)
        self.config = config
        self.joints = [
            "left.joint_0",
            "left.joint_1",
            "left.joint_2",
            "left.joint_3",
            "left.joint_4",
            "left.joint_5",
            "left.gripper",
            "right.joint_0",
            "right.joint_1",
            "right.joint_2",
            "right.joint_3",
            "right.joint_4",
            "right.joint_5",
            "right.gripper",
        ]
        # arms are folded up and rotated 
        self.joint_pos_sleep_l = [0.0] * 7
        self.joint_pos_sleep_r = [0.0] * 7
        # arms are folded up and rotated inwards 0.2 radians
        self.joint_pos_ready_l = [1.5708 - 0.3] + [0.0] * 6
        self.joint_pos_ready_r = [1.5708 + 0.3] + [0.0] * 6
        self.driver_l = None
        self.driver_r = None
        self.cameras = make_cameras_from_configs(config.cameras)

    def _set_all_positions(self, joint_pos_l: list[float], joint_pos_r: list[float], goal_time: float = 0.1, blocking: bool = True) -> None:
        self.driver_l.set_all_positions(
            trossen_arm.VectorDouble(joint_pos_l),
            goal_time=goal_time,
            blocking=blocking,
        )
        self.driver_r.set_all_positions(
            trossen_arm.VectorDouble(joint_pos_r),
            goal_time=goal_time,
            blocking=blocking,
        )

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.joints}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.driver_l is not None and self.driver_r is not None and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            self.driver_l = trossen_arm.TrossenArmDriver()
            self.driver_l.configure(
                trossen_arm.Model.wxai_v0,
                trossen_arm.StandardEndEffector.wxai_v0_leader,
                self.config.ip_address_l,
                False, # clear_error
            )
            self.driver_l.set_all_modes(trossen_arm.Mode.position)
            self.driver_r = trossen_arm.TrossenArmDriver()
            self.driver_r.configure(
                trossen_arm.Model.wxai_v0,
                trossen_arm.StandardEndEffector.wxai_v0_follower,
                self.config.ip_address_r,
                False, # clear_error
            )
            self.driver_r.set_all_modes(trossen_arm.Mode.position)
            if self.config.ready_on_connect:
                self._set_all_positions(self.joint_pos_ready_l, self.joint_pos_ready_r, self.config.goal_time_ready_sleep, True)
        except Exception as e:
            logger.error(f"Failed to connect to {self}: {e}")
            self.driver_l = None
            self.driver_r = None

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        joint_pos_l = self.driver_l.get_all_positions()
        joint_pos_r = self.driver_r.get_all_positions()
        obs_dict = {}
        for i, joint in enumerate(self.joints[:7]):
            obs_dict[f"{joint}.pos"] = joint_pos_l[i]
        for i, joint in enumerate(self.joints[7:]):
            obs_dict[f"{joint}.pos"] = joint_pos_r[i]
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any], goal_time: float = None, blocking: bool = False) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
    
        if goal_time is None:
            goal_time = self.config.goal_time_action

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        joint_pos_l = [goal_pos[joint] for joint in self.joints[:7]]
        joint_pos_r = [goal_pos[joint] for joint in self.joints[7:]]
        self._set_all_positions(joint_pos_l, joint_pos_r, goal_time, blocking)
        return {f"{joint}.pos": val for joint, val in goal_pos.items()}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self.config.sleep_on_disconnect:
            logger.info(f"{self} going to ready position.")
            self._set_all_positions(self.joint_pos_ready_l, self.joint_pos_ready_r, self.config.goal_time_ready_sleep, True)
            logger.info(f"{self} going to sleep position.")
            self._set_all_positions(self.joint_pos_sleep_l, self.joint_pos_sleep_r, self.config.goal_time_ready_sleep, True)
        if self.config.disable_torque_on_disconnect:
            logger.info(f"{self} disabling motor torques.")
            self.driver_l.set_all_modes(trossen_arm.Mode.idle)
            self.driver_r.set_all_modes(trossen_arm.Mode.idle)
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info(f"{self} disconnected.")
