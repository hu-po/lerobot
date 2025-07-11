import logging
import time
from functools import cached_property
from typing import Any
import os

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
        self.arm_l = None
        self.arm_r = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self.cond_cameras = make_cameras_from_configs(config.cond_cameras)

    def _connect_l(self, clear_error: bool = True) -> None:
        try:
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
            logger.debug(f"🦾 Loading right arm config from {config_filepath}")
            self.arm_r.load_configs_from_file(config_filepath)
            self.arm_r.set_all_modes(trossen_arm.Mode.position)
            self._set_positions_r(self.config.home_pos_r, self.config.goal_time_slow)
        except Exception as e:
            logger.warning(f"🦾❌ Failed to connect to {self} right arm:\n{e}")
            self.arm_r = None
        logger.info(f"✅🦾 {self} right arm connected.")

    def _get_positions_l(self) -> list[float]:
        if self.arm_l is None:
            logger.warning(f"🦾❌ Left arm is not connected.")
            return self.config.home_pos_l
        try:
            return self.arm_l.get_all_positions()
        except Exception as e:
            logger.warning(f"🦾❌ Failed to get left arm positions:\n{e}")
            return self.config.home_pos_l
    
    def _get_positions_r(self) -> list[float]:
        if self.arm_r is None:
            logger.warning(f"🦾❌ Right arm is not connected.")
            return self.config.home_pos_r
        try:
            return self.arm_r.get_all_positions()
        except Exception as e:
            logger.warning(f"🦾❌ Failed to get right arm positions:\n{e}")
            return self.config.home_pos_r

    def _set_positions_l(self, joints: list[float], goal_time: float = 1.0, block: bool = True) -> None:
        if self.arm_l is None:
            logger.warning(f"🦾❌ Left arm is not connected.")
            return
        try:
            logger.debug(f"🦾 Setting left arm positions: {joints}, goal_time: {goal_time}")
            if len(joints) != 7:
                logger.warning(f"🦾❌ Left arm positions length mismatch: {len(joints)} != 7")
                joints = joints[:7]
            self.arm_l.set_all_positions(
                trossen_arm.VectorDouble(joints),
                goal_time=goal_time,
                blocking=block,
            )
            read_joints = self._get_positions_l()
            mismatch: bool = False
            for i, joint in enumerate(self.joints[:7]):
                delta: float = abs(read_joints[i] - joints[i])
                if delta > self.config.joint_tolerance_warning:
                    logger.warning(f"🦾⚠️ Left arm position mismatch: {joint} {read_joints[i]} {joints[i]}")
                if delta > self.config.joint_tolerance_error:
                    logger.error(f"🦾❌ Left arm position mismatch: {joint} {read_joints[i]} {joints[i]}")
                    mismatch = True
            if mismatch:
                raise ValueError("left arm joints mismatch")
        except Exception as e:
            logger.warning(f"🦾❌ Failed to set left arm positions: \n{type(e)}:\n{e}\n{self._get_error_str_l()}")

    def _set_positions_r(self, joints: list[float], goal_time: float = 1.0, block: bool = True) -> None:
        if self.arm_r is None:
            logger.warning(f"🦾❌ Right arm is not connected.")
            return
        try:
            logger.debug(f"🦾 Setting right arm positions: {joints}, goal_time: {goal_time}")
            if len(joints) != 7:
                logger.warning(f"🦾❌ Right arm positions length mismatch: {len(joints)} != 7")
                joints = joints[:7]
            self.arm_r.set_all_positions(
                trossen_arm.VectorDouble(joints),
                goal_time=goal_time,
                blocking=block,
            )
            read_joints = self._get_positions_r()
            mismatch: bool = False
            for i, joint in enumerate(self.joints[7:]):
                delta: float = abs(read_joints[i] - joints[i])
                if delta > self.config.joint_tolerance_warning:
                    logger.warning(f"🦾⚠️ Right arm position mismatch: {joint} {read_joints[i]} {joints[i]}")
                if delta > self.config.joint_tolerance_error:
                    logger.error(f"🦾❌ Right arm position mismatch: {joint} {read_joints[i]} {joints[i]}")
                    mismatch = True
            if mismatch:
                raise ValueError("right arm joints mismatch")
        except Exception as e:
            logger.warning(f"🦾❌ Failed to set right arm positions: \n{type(e)}:\n{e}\n{self._get_error_str_r()}")

    # TODO: cartesian control?
    # # https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
    # # https://docs.trossenrobotics.com/trossen_arm/main/api/classtrossen__arm_1_1TrossenArmDriver.html
    # def _set_cartesian_l(
    #     self,
    #     goal_positions: list[float],
    #     interpolation_space: str = "cartesian",
    #     goal_time: float = 2.0,
    #     block: bool = True,
    # ) -> None:
    #     if self.arm_l is None:
    #         logger.warning("🦾❌ Left arm is not connected.")
    #         return
    #     try:
    #         from scipy.spatial.transform import Rotation as R
    #         rotvec = R.from_quat([goal_positions[3], goal_positions[4], goal_positions[5], goal_positions[6]]).as_rotvec()
    #         goal_positions[3:] = rotvec
    #         interp_space = getattr(trossen_arm.InterpolationSpace, interpolation_space)
    #         self.arm_l.set_cartesian_positions(
    #             trossen_arm.VectorDouble(goal_positions),
    #             interp_space,
    #             goal_time,
    #             block,
    #         )
    #     except Exception as e:
    #         logger.warning(f"🦾❌ Failed to set left arm cartesian positions: \n{type(e)}:\n{e}")

    # def _set_cartesian_r(
    #     self,
    #     goal_positions: list[float],
    #     interpolation_space: str = "cartesian",
    #     goal_time: float = 2.0,
    #     block: bool = True,
    # ) -> None:
    #     if self.arm_r is None:
    #         logger.warning("🦾❌ Right arm is not connected.")
    #         return
    #     try:
    #         interp_space = getattr(trossen_arm.InterpolationSpace, interpolation_space)
    #         self.arm_r.set_cartesian_positions(
    #             trossen_arm.VectorDouble(goal_positions),
    #             interp_space,
    #             goal_time,
    #             block,
    #         )
    #     except Exception as e:
    #         logger.warning(f"🦾❌ Failed to set right arm cartesian positions: \n{type(e)}:\n{e}")

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
        for i, joint in enumerate(self.joints[:7]):
            obs_dict[f"{joint}.pos"] = joint_pos_l[i]
        for i, joint in enumerate(self.joints[7:]):
            obs_dict[f"{joint}.pos"] = joint_pos_r[i]
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            try:
                obs_dict[cam_key] = cam.async_read()
            except Exception as e:
                logger.warning(f"❌🎥 Failed to read {cam_key}:\n{e}")
                obs_dict[cam_key] = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

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
        logger.debug(f"🤖 Action: {_action}")
        return _action
        
    def send_action(self, action: dict[str, Any], goal_time: float = None, block: str = "both") -> dict[str, Any]:
        if not self.is_connected:
            logger.warning(f"❌🤖 {self} is not connected.")
            # raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_time = self.config.goal_time_fast if goal_time is None else goal_time
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        joint_pos_r = [goal_pos[joint] for joint in self.joints[7:]]
        _block_right: bool = block == "right" or block == "both"
        self._set_positions_r(joint_pos_r, goal_time, block=_block_right)
        joint_pos_l = [goal_pos[joint] for joint in self.joints[:7]]
        _block_left: bool = block == "left" or block == "both"
        self._set_positions_l(joint_pos_l, goal_time, block=_block_left)
        return {f"{joint}.pos": val for joint, val in goal_pos.items()}

    def get_conditioning(self) -> dict[str, Any]:
        logger.debug(f"🤖🎥 {self} performing conditioning...")
        # connect conditioning cameras
        for cam in self.cond_cameras.values():
            try:
                cam.connect()
            except Exception as e:
                logger.warning(f"🎥❌Failed to connect to conditioning camera: {cam}: \n{e}")
        obs_dict = {}
        # read conditioning cameras
        for cam_key, cam in self.cond_cameras.items():
            start = time.perf_counter()
            try:
                obs_dict[cam_key] = cam.async_read()
            except Exception as e:
                logger.warning(f"❌🎥 Failed to read {cam_key}:\n{e}")
                obs_dict[cam_key] = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        # also add normal cameras to conditioning information
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            try:
                obs_dict[cam_key] = cam.async_read()
            except Exception as e:
                logger.warning(f"❌🎥 Failed to read {cam_key}:\n{e}")
                obs_dict[cam_key] = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
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
