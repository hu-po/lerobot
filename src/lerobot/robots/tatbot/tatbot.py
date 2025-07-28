import logging
from functools import cached_property
from typing import Any
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        self.home_pos_full = [*self.config.home_pos_l, *self.config.home_pos_r]
        
        max_workers = self.config.max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tatbot")

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

    def _configure_trossen_arm_logging(self) -> None:
        """Configure logging for trossen arm drivers."""
        try:
            # Get the log level from config
            log_level = getattr(logging, self.config.arm_log_level.upper(), logging.INFO)
            
            # Configure default logger
            default_logger = logging.getLogger(trossen_arm.TrossenArmDriver.get_default_logger_name())
            default_logger.setLevel(log_level)
            
            # Add handlers if not already present
            if not default_logger.handlers:
                default_logger.addHandler(logging.StreamHandler())
            
            # Configure left arm logger
            left_logger = logging.getLogger(
                trossen_arm.TrossenArmDriver.get_logger_name(
                    trossen_arm.Model.wxai_v0,
                    self.config.ip_address_l
                )
            )
            left_logger.setLevel(log_level)
            if not left_logger.handlers:
                left_logger.addHandler(logging.StreamHandler())
            
            # Configure right arm logger
            right_logger = logging.getLogger(
                trossen_arm.TrossenArmDriver.get_logger_name(
                    trossen_arm.Model.wxai_v0,
                    self.config.ip_address_r
                )
            )
            right_logger.setLevel(log_level)
            if not right_logger.handlers:
                right_logger.addHandler(logging.StreamHandler())
                
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🦾 Configured trossen arm logging with level: {self.config.arm_log_level}")
        except Exception as e:
            logger.warning(f"🦾❌ Failed to configure trossen arm logging:\n{e}")

    def _connect_arm(self, side: str, clear_error: bool = True) -> None:
        """Connects to a single arm (left or right)."""
        if side not in ("left", "right"):
            raise ValueError(f"Invalid side: {side}")

        ip_address = self.config.ip_address_l if side == "left" else self.config.ip_address_r
        config_filepath = self.config.arm_l_config_filepath if side == "left" else self.config.arm_r_config_filepath
        home_pos = self.config.home_pos_l if side == "left" else self.config.home_pos_r
        
        try:
            logger.debug(f"🦾 Connecting to {self} {side} arm")
            arm = trossen_arm.TrossenArmDriver()
            arm.configure(
                trossen_arm.Model.wxai_v0,
                trossen_arm.StandardEndEffector.wxai_v0_base,
                ip_address,
                clear_error,
                timeout=self.config.connection_timeout,
            )
            expanded_config_filepath = os.path.expanduser(config_filepath)
            logger.debug(f"🦾 Loading {side} arm config from {expanded_config_filepath}")
            arm.load_configs_from_file(expanded_config_filepath)
            arm.set_all_modes(trossen_arm.Mode.position)

            # Set arm attribute (self.arm_l or self.arm_r)
            setattr(self, f"arm_{side[0]}", arm)
            
            # Use the corresponding _set_positions method
            if side == "left":
                self._set_positions_l(home_pos, self.config.goal_time)
            else:
                self._set_positions_r(home_pos, self.config.goal_time)

            logger.info(f"✅🦾 {self} {side} arm connected.")
        except Exception:
            logger.exception(f"🦾❌ Failed to connect to {self} {side} arm")
            setattr(self, f"arm_{side[0]}", None)

    def _connect_arms(self, clear_error: bool = True) -> None:
        self._configure_trossen_arm_logging()
        futures = [
            self._executor.submit(self._connect_arm, "left", clear_error),
            self._executor.submit(self._connect_arm, "right", clear_error),
        ]
        # Wait for completion and re-raise any exceptions
        for future in as_completed(futures):
            future.result()

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

    def _set_positions(self, driver_handle, joints: list[float], goal_time: float, label: str, get_error_str_func, block: bool = False) -> None:
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

    def _set_positions_l(self, joints: list[float], goal_time: float, block: bool = False) -> None:
        self._set_positions(self.arm_l, joints, goal_time, "left", self._get_error_str_l, block)

    def _set_positions_r(self, joints: list[float], goal_time: float, block: bool = False) -> None:
        self._set_positions(self.arm_r, joints, goal_time, "right", self._get_error_str_r, block)

    def _get_error_str(self, driver_handle, label: str) -> str | None:
        if driver_handle is None:
            logger.warning(f"🦾❌ {label} arm is not connected.")
            return None
        try:
            return driver_handle.get_error_information()
        except Exception as e:
            logger.warning(f"🦾❌ Failed to get {label} arm error:\n{e}")
            return None

    def _get_error_str_l(self) -> str | None:
        return self._get_error_str(self.arm_l, "left")
        
    def _get_error_str_r(self) -> str | None:
        return self._get_error_str(self.arm_r, "right")

    def _for_each_camera(self, fn_name: str) -> None:
        """Helper to run a method on each camera concurrently."""
        all_cameras = list(self.rs_cameras.values()) + list(self.ip_cameras.values())
        
        def cam_op(camera):
            try:
                getattr(camera, fn_name)()
            except Exception as e:
                logger.warning(f"🎥❌ Camera {fn_name} failed for {camera}: {e}")

        # map consumes the iterator, ensuring all tasks run and complete
        self._executor.map(cam_op, all_cameras)

    def _connect_cameras(self) -> None:
        self._for_each_camera("connect")

    def _disconnect_cameras(self) -> None:
        self._for_each_camera("disconnect")

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
        self._connect_cameras()
        self._connect_arms()
        self.configure()
        logger.info(f"✅🤖 {self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _read_frame(self, cam_key: str, cam: Any) -> tuple[str, np.ndarray]:
        """Reads a frame from a single camera, handling errors gracefully."""
        try:
            frame = cam.async_read()
        except Exception as e:
            logger.warning(f"❌🎥 Failed to read {cam_key}: {e}")
            frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
        return cam_key, frame

    def get_observation(self, full: bool = False) -> dict[str, Any]:
        if not self.is_connected:
            logger.warning(f"❌🤖 {self} is not connected.")

        joint_pos_l = self._get_positions_l()
        joint_pos_r = self._get_positions_r()
        obs_dict = {f"{j}.pos": v for j, v in zip(self.joints, joint_pos_l + joint_pos_r)}

        cameras_to_read = self.rs_cameras.items()
        if full:
            cameras_to_read = list(self.rs_cameras.items()) + list(self.ip_cameras.items())

        futures = {
            self._executor.submit(self._read_frame, key, cam): key for key, cam in cameras_to_read
        }

        camera_frames = {}
        for future in as_completed(futures):
            key, frame = future.result()
            camera_frames[key] = frame
        
        obs_dict.update(camera_frames)
        return obs_dict

    def send_action(self, action: dict[str, Any], goal_time: float = None, safe: bool = False) -> dict[str, Any]:
        if not self.is_connected:
            logger.warning(f"❌🤖 {self} is not connected.")

        goal_time = self.config.goal_time if goal_time is None else goal_time
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        
        # Prepare joint positions
        goal_pos_l = [goal_pos[joint] for joint in self.joints[:7]]
        goal_pos_r = [goal_pos[joint] for joint in self.joints[7:]]

        # For non-blocking, just submit the tasks. They will run in the background.
        future_l = self._executor.submit(self._set_positions_l, goal_pos_l, goal_time, block=safe)
        future_r = self._executor.submit(self._set_positions_r, goal_pos_r, goal_time, block=safe)

        if safe:
            for future in as_completed([future_l, future_r]):
                future.result()

            for i, (_goal_pos, _curr_pos) in enumerate(zip(goal_pos_l, self._get_positions_l())):
                delta = abs(_goal_pos - _curr_pos)
                if delta > self.config.joint_tolerance_warning:
                    logger.warning(f"🦾⚠️ Left arm joint position {self.joints[i]} mismatch: {_goal_pos} != {_curr_pos}")
                if delta > self.config.joint_tolerance_error:
                    logger.error(f"🦾❌ Left arm joint position {self.joints[i]} mismatch: {_goal_pos} != {_curr_pos}")
                    raise ValueError("Left arm joints mismatch")

            for i, (_goal_pos, _curr_pos) in enumerate(zip(goal_pos_r, self._get_positions_r())):
                delta = abs(_goal_pos - _curr_pos)
                if delta > self.config.joint_tolerance_warning:
                    logger.warning(f"🦾⚠️ Right arm joint position {self.joints[i]} mismatch: {_goal_pos} != {_curr_pos}")
                if delta > self.config.joint_tolerance_error:
                    logger.error(f"🦾❌ Right arm joint position {self.joints[i]} mismatch: {_goal_pos} != {_curr_pos}")
                    raise ValueError("Right arm joints mismatch")
            
        return {f"{joint}.pos": val for joint, val in goal_pos.items()}

    def disconnect(self):
        if not self.is_connected:
            logger.warning(f"❌🤖 {self} is not connected.")

        # first try and shutdown the arms gracefully
        if self.arm_l is not None:
            self.arm_l.cleanup()
        if self.arm_r is not None:
            self.arm_r.cleanup()

        # then try and get the error strings
        error_str_l = self._get_error_str_l()
        if error_str_l is not None:
            logger.error(f"🦾❌ Left arm error: {error_str_l}")
        error_str_r = self._get_error_str_r()
        if error_str_r is not None:
            logger.error(f"🦾❌ Right arm error: {error_str_r}")

        # re-connect and send to home pose
        self._connect_arms(clear_error=True)
        self.send_action(self._urdf_joints_to_action(self.home_pos_full), safe=True)

        # set arms to idle
        if self.arm_l is not None:
            self.arm_l.set_all_modes(trossen_arm.Mode.idle)
            logger.info(f"✅🦾 {self} left arm idle.")
        if self.arm_r is not None:
            self.arm_r.set_all_modes(trossen_arm.Mode.idle)
            logger.info(f"✅🦾 {self} right arm idle.")

        self._disconnect_cameras()

        logger.info(f"🤖 Shutting down thread pool...")
        self._executor.shutdown(wait=False)
        logger.info(f"✅🤖 {self} disconnected and thread pool shut down.")
