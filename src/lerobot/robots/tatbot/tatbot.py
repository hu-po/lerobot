import logging
import time
from functools import cached_property
from typing import Any
import os
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
from lerobot.cameras.utils import make_cameras_from_configs
# from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

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
        self.cameras = make_cameras_from_configs(config.cameras)
        self.cond_cameras = make_cameras_from_configs(config.cond_cameras)

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
            self._set_positions_l(self.config.home_pos_l, self.config.goal_time_slow)
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
            self._set_positions_r(self.config.home_pos_r, self.config.goal_time_slow)
        except Exception as e:
            logger.warning(f"🦾❌ Failed to connect to {self} right arm:\n{e}")
            self.arm_r = None
        logger.info(f"✅🦾 {self} right arm connected.")

    def _get_positions(self, driver_handle, fallback_pose: list[float], label: str) -> list[float]:
        if driver_handle is None:
            logger.warning(f"🦾❌ {label} arm is not connected.")
            return fallback_pose
        try:
            return list(driver_handle.get_all_positions())
        except Exception as e:
            logger.warning(f"🦾❌ Failed to get {label} arm positions:\n{e}")
            return fallback_pose
    
    def _get_positions_l(self) -> list[float]:
        return self._get_positions(self.arm_l, self.config.home_pos_l, "Left")
    
    def _get_positions_r(self) -> list[float]:
        return self._get_positions(self.arm_r, self.config.home_pos_r, "Right")

    def _set_positions(self, driver_handle, joints: list[float], goal_time: float, block: bool, label: str, get_error_str_func) -> None:
        if driver_handle is None:
            logger.warning(f"🦾❌ {label} arm is not connected.")
            return
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🦾 Setting {label.lower()} arm positions: {joints}, goal_time: {goal_time}")
            if len(joints) != 7:
                raise ValueError(f"🦾❌ {label} arm positions length mismatch: {len(joints)} != 7")
            driver_handle.set_all_positions(
                trossen_arm.VectorDouble(joints),
                goal_time=goal_time,
                blocking=block,
            )
        except Exception as e:
            logger.warning(f"🦾❌ Failed to set {label.lower()} arm positions: \n{type(e)}:\n{e}\n{get_error_str_func()}")

    def _set_positions_l(self, joints: list[float], goal_time: float = 1.0, block: bool = True) -> None:
        self._set_positions(self.arm_l, joints, goal_time, block, "Left", self._get_error_str_l)

    def _set_positions_r(self, joints: list[float], goal_time: float = 1.0, block: bool = True) -> None:
        self._set_positions(self.arm_r, joints, goal_time, block, "Right", self._get_error_str_r)

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
    def _motors_ft(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.joints}
    
    # @property
    # def _ee_pose_ft(self) -> dict[str, type]:
    #     return {dim: float for dim in ["x", "y", "z", "qw", "qx", "qy", "qz"]}

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
        # return {**self._motors_ft, **self._ee_pose_ft}

    @property
    def is_connected(self) -> bool:
        return self.arm_l is not None and self.arm_r is not None and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            logger.info(f"✅🤖 {self} already connected.")
            return
            # raise DeviceAlreadyConnectedError(f"❌🤖 {self} already connected")

        for cam in self.cameras.values():
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
            # raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        joint_pos_l = self._get_positions_l()
        joint_pos_r = self._get_positions_r()
        obs_dict = {}
        obs_dict.update({f"{j}.pos": v for j, v in zip(self.joints, joint_pos_l + joint_pos_r)})
        dt_ms = (time.perf_counter() - start) * 1e3
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras in parallel
        def read_camera(cam_key, cam):
            start = time.perf_counter()
            try:
                frame = cam.async_read()
            except Exception as e:
                logger.warning(f"❌🎥 Failed to read {cam_key}:\n{e}")
                frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
            dt_ms = (time.perf_counter() - start) * 1e3
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
            return cam_key, frame
        
        with ThreadPoolExecutor(max_workers=max(1, len(self.cameras))) as executor:
            futures = [executor.submit(read_camera, cam_key, cam) for cam_key, cam in self.cameras.items()]
            for future in futures:
                cam_key, frame = future.result()
                obs_dict[cam_key] = frame

        return obs_dict

    def _urdf_joints_to_action(self, urdf_joints: list[float]) -> dict[str, float]:
        _action = {
            "left.joint_0.pos": urdf_joints[0],
            "left.joint_1.pos": urdf_joints[1],
            "left.joint_2.pos": urdf_joints[2],
            "left.joint_3.pos": urdf_joints[3],
            "left.joint_4.pos": urdf_joints[4],
            "left.joint_5.pos": urdf_joints[5],
            "left.gripper.pos": urdf_joints[6],
            "right.joint_0.pos": urdf_joints[8],
            "right.joint_1.pos": urdf_joints[9],
            "right.joint_2.pos": urdf_joints[10],
            "right.joint_3.pos": urdf_joints[11],
            "right.joint_4.pos": urdf_joints[12],
            "right.joint_5.pos": urdf_joints[13],
            "right.gripper.pos": urdf_joints[14],
        }
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🤖 Action: {_action}")
        return _action
        
    def _wait_for_arms(self, timeout: float | None = None, goal_pos_l: list[float] = None, goal_pos_r: list[float] = None, wait_left: bool = True, wait_right: bool = True) -> None:
        """Wait for arms to complete their movements.
        
        Args:
            timeout: Maximum time to wait. If None, uses goal_time_slow
            goal_pos_l: Target positions for left arm for validation
            goal_pos_r: Target positions for right arm for validation  
            wait_left: Whether to wait for left arm completion
            wait_right: Whether to wait for right arm completion
        """
        # If validation is disabled, return immediately to avoid unnecessary polling
        if not self.config.validate_positions:
            return
            
        if timeout is None:
            timeout = self.config.goal_time_slow
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < timeout:
            # Check if arms have reached their target positions
            if wait_left and self.arm_l is not None:
                joint_pos_l = self._get_positions_l()
            if wait_right and self.arm_r is not None:
                joint_pos_r = self._get_positions_r()
            
            # Check completion if validation is enabled and goals provided
            all_complete = True
            if wait_left and goal_pos_l is not None and self.arm_l is not None:
                deltas_l = np.abs(np.array(joint_pos_l) - np.array(goal_pos_l))
                if np.any(deltas_l > self.config.joint_tolerance_error):
                    all_complete = False
            if wait_right and goal_pos_r is not None and self.arm_r is not None:
                deltas_r = np.abs(np.array(joint_pos_r) - np.array(goal_pos_r))
                if np.any(deltas_r > self.config.joint_tolerance_error):
                    all_complete = False
            if all_complete:
                break
            
            time.sleep(0.1)  # Increased sleep to reduce busy-wait overhead
        
        # Log warning if timeout reached without completion
        if time.perf_counter() - start_time >= timeout:
            logger.warning(f"🦾⚠️ Arm movement timeout after {timeout:.1f}s")
        
        # Validate positions if enabled
        self._validate_arm_positions(goal_pos_l, goal_pos_r, wait_left, wait_right)
    
    def _validate_arm_positions(self, goal_pos_l: list[float] = None, goal_pos_r: list[float] = None, validate_left: bool = True, validate_right: bool = True) -> None:
        """Validate arm positions against target positions."""
        if validate_left and goal_pos_l is not None and self.arm_l is not None:
            joint_pos_l = self._get_positions_l()
            deltas_l = np.abs(np.array(joint_pos_l) - np.array(goal_pos_l))
            for i, joint in enumerate(self.joints[:7]):
                delta = deltas_l[i]
                if delta > self.config.joint_tolerance_warning:
                    logger.warning(f"🦾⚠️ Left arm position mismatch: {joint} {joint_pos_l[i]} {goal_pos_l[i]}")
                if delta > self.config.joint_tolerance_error:
                    logger.error(f"🦾❌ Left arm position mismatch: {joint} {joint_pos_l[i]} {goal_pos_l[i]}")
                    raise ValueError("Left arm joints mismatch")
        
        if validate_right and goal_pos_r is not None and self.arm_r is not None:
            joint_pos_r = self._get_positions_r()
            deltas_r = np.abs(np.array(joint_pos_r) - np.array(goal_pos_r))
            for i, joint in enumerate(self.joints[7:]):
                delta = deltas_r[i]
                if delta > self.config.joint_tolerance_warning:
                    logger.warning(f"🦾⚠️ Right arm position mismatch: {joint} {joint_pos_r[i]} {goal_pos_r[i]}")
                if delta > self.config.joint_tolerance_error:
                    logger.error(f"🦾❌ Right arm position mismatch: {joint} {joint_pos_r[i]} {goal_pos_r[i]}")
                    raise ValueError("Right arm joints mismatch")

    def send_action(self, action: dict[str, Any], goal_time: float = None, block: str = "both") -> dict[str, Any]:
        if not self.is_connected:
            logger.warning(f"❌🤖 {self} is not connected.")
            # raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_time = self.config.goal_time_fast if goal_time is None else goal_time
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        
        # Prepare joint positions
        joint_pos_l = [goal_pos[joint] for joint in self.joints[:7]]
        joint_pos_r = [goal_pos[joint] for joint in self.joints[7:]]
        
        # Issue both arm commands in parallel using threads
        def set_left_arm():
            self._set_positions_l(joint_pos_l, goal_time, block=False)
        
        def set_right_arm():
            self._set_positions_r(joint_pos_r, goal_time, block=False)
        
        left_thread = threading.Thread(target=set_left_arm)
        right_thread = threading.Thread(target=set_right_arm)
        
        left_thread.start()
        right_thread.start()
        left_thread.join()
        right_thread.join()
        
        # Wait for completion if requested
        if block != "none":
            wait_left = block in ["both", "left"]
            wait_right = block in ["both", "right"]
            self._wait_for_arms(
                goal_time, 
                goal_pos_l=joint_pos_l if wait_left else None,
                goal_pos_r=joint_pos_r if wait_right else None,
                wait_left=wait_left,
                wait_right=wait_right
            )
        
        return {f"{joint}.pos": val for joint, val in goal_pos.items()}

    def get_conditioning(self) -> dict[str, Any]:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🤖🎥 {self} performing conditioning...")
        # connect conditioning cameras
        for cam in self.cond_cameras.values():
            try:
                cam.connect()
            except Exception as e:
                logger.warning(f"🎥❌Failed to connect to conditioning camera: {cam}: \n{e}")
        obs_dict = {}
        # read conditioning cameras in parallel
        def read_cond_camera(cam_key, cam):
            start = time.perf_counter()
            try:
                frame = cam.async_read()
            except Exception as e:
                logger.warning(f"❌🎥 Failed to read {cam_key}:\n{e}")
                frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
            dt_ms = (time.perf_counter() - start) * 1e3
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
            return cam_key, frame
        
        # Read conditioning cameras in parallel
        with ThreadPoolExecutor(max_workers=max(1, len(self.cond_cameras))) as executor:
            futures = [executor.submit(read_cond_camera, cam_key, cam) for cam_key, cam in self.cond_cameras.items()]
            for future in futures:
                cam_key, frame = future.result()
                obs_dict[cam_key] = frame
        
        # also add normal cameras to conditioning information in parallel
        with ThreadPoolExecutor(max_workers=max(1, len(self.cameras))) as executor:
            futures = [executor.submit(read_cond_camera, cam_key, cam) for cam_key, cam in self.cameras.items()]
            for future in futures:
                cam_key, frame = future.result()
                obs_dict[cam_key] = frame
        # disconnect conditioning cameras
        for cam in self.cond_cameras.values():
            try:
                cam.disconnect()
            except Exception as e:
                logger.warning(f"🎥❌ Failed to disconnect from {cam}:\n{e}")
        return obs_dict

    def disconnect(self):
        if not self.is_connected:
            logger.warning(f"❌🤖 {self} is not connected.")
            # raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info(f"🤖 {self} going to home position.")
        self._set_positions_l(self.config.home_pos_l, goal_time=self.config.goal_time_slow)
        self._set_positions_r(self.config.home_pos_r, goal_time=self.config.goal_time_slow)

        if self.arm_l is not None:
            try:
                self.arm_l.set_all_modes(trossen_arm.Mode.idle)
                logger.info(f"✅🦾 left arm idle.")
            except Exception as e:
                logger.warning(f"🦾❌ Failed to idle left arm:\n{e}")

        if self.arm_r is not None:
            try:
                self.arm_r.set_all_modes(trossen_arm.Mode.idle)
                logger.info(f"✅🦾 right arm idle.")
            except Exception as e:
                logger.warning(f"🦾❌ Failed to idle right arm:\n{e}")

        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception as e:
                logger.warning(f"🎥❌ Failed to disconnect from {cam}:\n{e}")

        for cam in self.cond_cameras.values():
            try:
                cam.disconnect()
            except Exception as e:
                logger.warning(f"🎥❌ Failed to disconnect from {cam}:\n{e}")

        logger.info(f"✅🤖 {self} disconnected.")
