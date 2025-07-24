import logging
import time
from functools import cached_property
from typing import Any
import os
import threading

import numpy as np
from lerobot.cameras.utils import make_cameras_from_configs

import trossen_arm

from ..robot import Robot
from .config_tatbot import TatbotConfig

logger = logging.getLogger(__name__)


class Tatbot(Robot):
    """
    A [tatbot](https://github.com/hu-po/tatbot) robot.
    """

    config_class = TatbotConfig
    name = "tatbot"

    def __init__(self, config: TatbotConfig):
        super().__init__(config)
        self.config = config
        self.joints = [f"{side}.{name}" for side in ("left","right") for name in ("joint_0","joint_1","joint_2","joint_3","joint_4","joint_5","gripper")]
        self.arm_l = None
        self.arm_r = None
        self.rs_cameras = make_cameras_from_configs(config.rs_cameras)
        self.ip_cameras = make_cameras_from_configs(config.ip_cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.joints}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        _ft = {}
        for cam_name in self.rs_cameras.keys():
            _ft[cam_name] = (self.config.rs_cameras[cam_name].height, self.config.rs_cameras[cam_name].width, 3)
        for cam_name in self.ip_cameras.keys():
            _ft[cam_name] = (self.config.ip_cameras[cam_name].height, self.config.ip_cameras[cam_name].width, 3)
        return _ft

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    def _urdf_joints_to_action(self, urdf_joints: list[float]) -> dict[str, float]:
        _action = {f"{joint}.pos": urdf_joints[i] for i, joint in enumerate(self.joints)}
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🤖 Action: {_action}")
        return _action

    def _connect_l(self, clear_error: bool = True) -> None:
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🦾 Connecting to {self} left arm")
            self.arm_l = trossen_arm.TrossenArmDriver()
            self.arm_l.configure(
                trossen_arm.Model.wxai_v0,
                trossen_arm.StandardEndEffector.wxai_v0_base,
                self.config.ip_address_l,
                clear_error,
                timeout=self.config.connection_timeout,
            )
            config_filepath = os.path.expanduser(self.config.arm_l_config_filepath)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🦾 Loading left arm config from {config_filepath}")
            self.arm_l.load_configs_from_file(config_filepath)
            self.arm_l.set_all_modes(trossen_arm.Mode.position)
            self._set_positions_l(self.config.home_pos_l, self.config.goal_time)
        except Exception as e:
            logger.warning(f"🦾❌ Failed to connect to {self} left arm:\n{e}")
            self.arm_l = None
        logger.info(f"✅🦾 {self} left arm connected.")

    def _connect_r(self, clear_error: bool = True) -> None:
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🦾 Connecting to {self} right arm")
            self.arm_r = trossen_arm.TrossenArmDriver()
            self.arm_r.configure(
                trossen_arm.Model.wxai_v0,
                trossen_arm.StandardEndEffector.wxai_v0_base,
                self.config.ip_address_r,
                clear_error,
                timeout=self.config.connection_timeout,
            )
            config_filepath = os.path.expanduser(self.config.arm_r_config_filepath)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🦾 Loading right arm config from {config_filepath}")
            self.arm_r.load_configs_from_file(config_filepath)
            self.arm_r.set_all_modes(trossen_arm.Mode.position)
            self._set_positions_r(self.config.home_pos_r, self.config.goal_time)
        except Exception as e:
            logger.warning(f"🦾❌ Failed to connect to {self} right arm:\n{e}")
            self.arm_r = None
        logger.info(f"✅🦾 {self} right arm connected.")

    def _get_positions(self, driver_handle, fallback_pose: list[float], label: str) -> list[float]:
        if driver_handle is None:
            logger.warning(f"🦾❌ {label} arm is not connected.")
            return fallback_pose
        try:
            return list(driver_handle.get_all_positions()[:7])
        except Exception as e:
            logger.warning(f"🦾❌ Failed to get {label} arm positions:\n{e}")
            return fallback_pose
    
    def _get_positions_l(self) -> list[float]:
        return self._get_positions(self.arm_l, self.config.home_pos_l, "left")
    
    def _get_positions_r(self) -> list[float]:
        return self._get_positions(self.arm_r, self.config.home_pos_r, "right")

    def _set_positions(self, driver_handle, joints: list[float], goal_time: float, block: bool, label: str, get_error_str_func) -> None:
        if driver_handle is None:
            logger.warning(f"🦾❌ {label} arm is not connected.")
            return
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🦾 Setting {label.lower()} arm positions: {joints}, goal_time: {goal_time}")
            if len(joints) != 7:
                raise ValueError(f"🦾❌ {label} arm positions length mismatch: {len(joints)} != 7")
            driver_handle.set_all_positions(trossen_arm.VectorDouble(joints), goal_time=goal_time, blocking=block)
        except Exception as e:
            logger.warning(f"🦾❌ Failed to set {label.lower()} arm positions: \n{type(e)}:\n{e}\n{get_error_str_func()}")

    def _set_positions_l(self, joints: list[float], goal_time: float, block: bool) -> None:
        self._set_positions(self.arm_l, joints, goal_time, block, "left", self._get_error_str_l)

    def _set_positions_r(self, joints: list[float], goal_time: float, block: bool) -> None:
        self._set_positions(self.arm_r, joints, goal_time, block, "right", self._get_error_str_r)

    def _get_error_str_l(self) -> str:
        if self.arm_l is None:
            logger.warning(f"🦾❌ Left arm is not connected.")
            return ""
        try:
            return self.arm_l.get_error_information()
        except Exception as e:
            logger.warning(f"🦾❌ Failed to get left arm error:\n{e}")
            return ""
        
    def _get_error_str_r(self) -> str:
        if self.arm_r is None:
            logger.warning(f"🦾❌ Right arm is not connected.")
            return ""
        try:
            return self.arm_r.get_error_information()
        except Exception as e:
            logger.warning(f"🦾❌ Failed to get right arm error:\n{e}")
            return ""

    @property
    def is_connected(self) -> bool:
        return self.arm_l is not None and \
            self.arm_r is not None and \
            all(cam.is_connected for cam in self.rs_cameras.values()) and \
            all(cam.is_connected for cam in self.ip_cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            logger.info(f"✅🤖 {self} already connected.")
            return

        for cam in self.rs_cameras.values():
            try:
                cam.connect()
            except Exception as e:
                logger.warning(f"🎥❌Failed to connect to camera: {cam}: \n{e}")
        for cam in self.ip_cameras.values():
            try:
                cam.connect()
            except Exception as e:
                logger.warning(f"🎥❌Failed to connect to camera: {cam}: \n{e}")
        self._connect_l()
        self._connect_r()
        self.configure()
        logger.info(f"✅🤖 {self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            logger.warning(f"❌🤖 {self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        joint_pos_l = self._get_positions_l()
        joint_pos_r = self._get_positions_r()
        obs_dict = {}
        obs_dict.update({f"{j}.pos": v for j, v in zip(self.joints, joint_pos_l + joint_pos_r)})
        dt_ms = (time.perf_counter() - start) * 1e3
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from rs cameras
        for cam_key, cam in self.rs_cameras.items():
            start = time.perf_counter()
            try:
                frame = cam.async_read()
            except Exception as e:
                logger.warning(f"❌🎥 Failed to read {cam_key}:\n{e}")
                frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
            dt_ms = (time.perf_counter() - start) * 1e3
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
            obs_dict[cam_key] = frame

        return obs_dict
    
    def get_observation_full(self) -> dict[str, Any]:
        obs_dict = self.get_observation()

        # Capture images from ip cameras
        for cam_key, cam in self.ip_cameras.items():
            start = time.perf_counter()
            try:
                frame = cam.async_read()
            except Exception as e:
                logger.warning(f"❌🎥 Failed to read {cam_key}:\n{e}")
                frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
            dt_ms = (time.perf_counter() - start) * 1e3
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
            obs_dict[cam_key] = frame
        return obs_dict

    def send_action(self, action: dict[str, Any], goal_time: float = None) -> dict[str, Any]:
        if not self.is_connected:
            logger.warning(f"❌🤖 {self} is not connected.")

        goal_time = self.config.goal_time if goal_time is None else goal_time
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        
        # Prepare joint positions
        goal_pos_l = [goal_pos[joint] for joint in self.joints[:7]]
        goal_pos_r = [goal_pos[joint] for joint in self.joints[7:]]
        
        # Issue both arm commands in parallel using threads
        def set_left_arm():
            self._set_positions_l(goal_pos_l, goal_time, block=True)
        
        def set_right_arm():
            self._set_positions_r(goal_pos_r, goal_time, block=True)
        
        left_thread = threading.Thread(target=set_left_arm)
        right_thread = threading.Thread(target=set_right_arm)
        
        left_thread.start()
        right_thread.start()
        left_thread.join()
        right_thread.join()
        
        if self.config.validate_positions:
            for i, (goal_pos, curr_pos) in enumerate(zip(goal_pos_l, self._get_positions_l())):
                delta = abs(goal_pos - curr_pos)
                if delta > self.config.joint_tolerance_warning:
                    logger.warning(f"🦾⚠️ Left arm joint position {self.joints[i]} mismatch: {goal_pos} != {curr_pos}")
                if delta > self.config.joint_tolerance_error:
                    logger.error(f"🦾❌ Left arm joint position {self.joints[i]} mismatch: {goal_pos} != {curr_pos}")
                    raise ValueError("Left arm joints mismatch")

            for i, (goal_pos, curr_pos) in enumerate(zip(goal_pos_r, self._get_positions_r())):
                delta = abs(goal_pos - curr_pos)
                if delta > self.config.joint_tolerance_warning:
                    logger.warning(f"🦾⚠️ Right arm joint position {self.joints[i]} mismatch: {goal_pos} != {curr_pos}")
                if delta > self.config.joint_tolerance_error:
                    logger.error(f"🦾❌ Right arm joint position {self.joints[i]} mismatch: {goal_pos} != {curr_pos}")
                    raise ValueError("Right arm joints mismatch")
        
        return {f"{joint}.pos": val for joint, val in goal_pos.items()}

    def disconnect(self):
        if not self.is_connected:
            logger.warning(f"❌🤖 {self} is not connected.")

        # first try and get the error strings
        self._connect_l(clear_error=False)
        logger.error(self._get_error_str_l())
        self._connect_r(clear_error=False)
        logger.error(self._get_error_str_r())

        # then clear errors and go to home positions
        logger.info(f"🤖 {self} left arm going to home position.")
        self._connect_l()
        self._set_positions_l(self.config.home_pos_l, self.config.goal_time, True)
        if self.arm_l is not None:
            self.arm_l.set_all_modes(trossen_arm.Mode.idle)
        logger.info(f"✅🦾 {self} left arm idle.")

        logger.info(f"🤖 {self} right arm going to home position.")
        self._connect_r()
        self._set_positions_r(self.config.home_pos_r, self.config.goal_time, True)
        if self.arm_r is not None:
            self.arm_r.set_all_modes(trossen_arm.Mode.idle)
        logger.info(f"✅🦾 {self} right arm idle.")

        # disconnect cameras
        for cam in self.rs_cameras.values():
            try:
                cam.disconnect()
            except Exception as e:
                logger.warning(f"🎥❌ Failed to disconnect from {cam}:\n{e}")

        for cam in self.ip_cameras.values():
            try:
                cam.disconnect()
            except Exception as e:
                logger.warning(f"🎥❌ Failed to disconnect from {cam}:\n{e}")

        logger.info(f"✅🤖 {self} disconnected.")
