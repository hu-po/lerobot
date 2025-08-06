import logging
from functools import cached_property
from typing import Any
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

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
        
        # Thread safety locks
        self._connect_lock = threading.Lock()
        self._disconnect_lock = threading.Lock()
        self._connected = False  # Start disconnected

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
        """
        Connects to the robot in a transactional manner with thread safety.
        If any part of the connection fails, it triggers a full disconnect
        to ensure the robot is left in a clean state.
        """
        with self._connect_lock:
            if self.is_connected:
                logger.info(f"✅🤖 {self} already connected.")
                return

            try:
                logger.info(f"🤖 Starting connection sequence for {self}")
                
                # Create fresh thread pool if needed (after disconnect)
                if self._executor is None:
                    logger.debug("🧵 Creating fresh thread pool for reconnection")
                    self._executor = ThreadPoolExecutor(
                        max_workers=self.config.max_workers, 
                        thread_name_prefix="tatbot"
                    )
                
                self._connect_cameras()
                self._connect_arms()
                self.configure()
                
                # Final check to ensure all components are truly responsive
                if not self._check_arm_responsiveness():
                    raise RuntimeError("Arms connected but not responsive.")
                    
                # Set connected flag on successful connection
                self._connected = True
                logger.info(f"✅🤖 {self} connected and responsive.")

            except Exception as e:
                logger.error(f"🤖❌ Connection failed: {e}")
                logger.error("🤖 Triggering full disconnect to ensure clean state...")
                try:
                    self.disconnect()
                except Exception as disconnect_error:
                    logger.error(f"🤖❌ Disconnect during failed connection also failed: {disconnect_error}")
                # Re-raise the exception so the caller knows the connection failed
                raise

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

    def send_action(self, action: dict[str, Any], goal_time: float = None, safe: bool = False, left_first: bool = True) -> dict[str, Any]:
        """
        Sends an action to the robot after proactively checking for arm responsiveness.
        Uses robust checks and fail-fast behavior for reliable operation.
        """
        # Use the robust responsiveness check instead of just is_connected
        if not self._check_arm_responsiveness():
            logger.error("🦾❌ Cannot send action: Arms are not responsive.")
            raise RuntimeError("Cannot send action to non-responsive arms.")

        goal_time = self.config.goal_time if goal_time is None else goal_time
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        
        # Prepare joint positions
        goal_pos_l = [goal_pos[joint] for joint in self.joints[:7]]
        goal_pos_r = [goal_pos[joint] for joint in self.joints[7:]]

        try:
            if left_first:
                future_l = self._executor.submit(self._set_positions_l, goal_pos_l, goal_time, block=safe)
                future_r = self._executor.submit(self._set_positions_r, goal_pos_r, goal_time, block=safe)
            else:
                future_r = self._executor.submit(self._set_positions_r, goal_pos_r, goal_time, block=safe)
                future_l = self._executor.submit(self._set_positions_l, goal_pos_l, goal_time, block=safe)

            if safe:
                # Wait for both arms to complete
                for future in as_completed([future_l, future_r]):
                    future.result()

                # Verify positions with detailed error reporting
                self._verify_action_completion(goal_pos_l, goal_pos_r)
                
        except Exception as e:
            logger.error(f"🦾❌ Action execution failed: {e}")
            # Re-check responsiveness after failure
            if not self._check_arm_responsiveness():
                logger.error("🦾❌ Arms became unresponsive during action execution")
            raise
            
        return {f"{joint}.pos": val for joint, val in goal_pos.items()}

    def _verify_action_completion(self, goal_pos_l: list[float], goal_pos_r: list[float]) -> None:
        """Verify that the commanded positions were actually reached."""
        # Check left arm positions
        current_pos_l = self._get_positions_l()
        for i, (goal_pos, curr_pos) in enumerate(zip(goal_pos_l, current_pos_l)):
            delta = abs(goal_pos - curr_pos)
            if delta > self.config.joint_tolerance_warning:
                logger.warning(f"🦾⚠️ Left arm joint {self.joints[i]} position mismatch: goal={goal_pos:.3f}, actual={curr_pos:.3f}, delta={delta:.3f}")
            if delta > self.config.joint_tolerance_error:
                logger.error(f"🦾❌ Left arm joint {self.joints[i]} position error: goal={goal_pos:.3f}, actual={curr_pos:.3f}, delta={delta:.3f}")
                raise ValueError(f"Left arm joint {self.joints[i]} position error exceeds tolerance")

        # Check right arm positions  
        current_pos_r = self._get_positions_r()
        for i, (goal_pos, curr_pos) in enumerate(zip(goal_pos_r, current_pos_r)):
            delta = abs(goal_pos - curr_pos)
            joint_idx = i + 7  # Right arm joints start at index 7
            if delta > self.config.joint_tolerance_warning:
                logger.warning(f"🦾⚠️ Right arm joint {self.joints[joint_idx]} position mismatch: goal={goal_pos:.3f}, actual={curr_pos:.3f}, delta={delta:.3f}")
            if delta > self.config.joint_tolerance_error:
                logger.error(f"🦾❌ Right arm joint {self.joints[joint_idx]} position error: goal={goal_pos:.3f}, actual={curr_pos:.3f}, delta={delta:.3f}")
                raise ValueError(f"Right arm joint {self.joints[joint_idx]} position error exceeds tolerance")

    def disconnect(self, reboot_controller: bool = False):
        """
        Thread-safe, re-entrant disconnect sequence following Trossen best practices.
        ALWAYS ensures arms return to sleep pose via reconnect if needed.
        Can be called multiple times safely and handles various error states.
        """
        with self._disconnect_lock:
            if not self._connected:
                logger.info(f"🤖 {self} already disconnected")
                return
            try:
                self._disconnect_impl(reboot_controller)
            finally:
                self._connected = False

    def _disconnect_impl(self, reboot_controller: bool = False):
        """Implementation of disconnect sequence."""
        logger.info(f"🤖 Starting robust disconnect sequence for {self}")
        start_time = time.time()
        
        try:
            # Phase 0: One clean reconnect attempt if link is already dead
            if not self._check_arm_responsiveness():
                logger.info("🦾 Link lost – trying one clean reconnect to park the arm")
                try:
                    self._connect_arms(clear_error=True)
                    time.sleep(self.config.reboot_wait_time)
                except Exception as e:
                    logger.warning(f"🦾⚠️ Initial reconnect failed: {e}")
            
            # Phase 1: Stop all motion immediately (safety first)
            self._emergency_stop_all_motion()
            
            # Phase 2: Ensure clean reconnection and return to sleep pose
            self._ensure_sleep_pose_via_reconnect()
            
            # Phase 3: Final cleanup and shutdown
            self._final_cleanup_and_shutdown(reboot_controller)
            
        except Exception as e:
            logger.error(f"🤖❌ Disconnect sequence failed: {e}")
            # Always try final emergency cleanup
            self._emergency_cleanup()
        
        elapsed = time.time() - start_time
        logger.info(f"✅🤖 {self} disconnect sequence completed in {elapsed:.1f}s")

    def _emergency_stop_all_motion(self):
        """Phase 1: Immediately stop all arm motion for safety."""
        logger.debug("🤖 Phase 1: Emergency stop all motion")
        
        def stop_arm_motion(arm, label, current_positions):
            if arm is not None:
                try:
                    # Hold current position instead of going limp for safety
                    logger.debug(f"🦾 Emergency stopping {label} arm motion")
                    arm.set_all_positions(
                        trossen_arm.VectorDouble(current_positions), 
                        goal_time=0.1,  # Very short time to hold position
                        blocking=True   # Wait for command to be accepted
                    )
                    logger.info(f"✅🦾 {label} arm motion stopped and holding position")
                except Exception as e:
                    logger.warning(f"🦾⚠️ Failed to stop {label} arm motion, falling back to idle: {e}")
                    try:
                        # Fallback to idle if position hold fails
                        arm.set_all_modes(trossen_arm.Mode.idle)
                    except Exception:
                        pass

        # Get current positions for holding
        current_pos_l = self._get_positions_l()
        current_pos_r = self._get_positions_r()
        
        # Stop both arms concurrently for maximum speed
        futures = []
        if self.arm_l is not None:
            futures.append(self._executor.submit(stop_arm_motion, self.arm_l, "Left", current_pos_l))
        if self.arm_r is not None:
            futures.append(self._executor.submit(stop_arm_motion, self.arm_r, "Right", current_pos_r))
        
        # Wait for all stops to complete with timeout
        try:
            for future in as_completed(futures, timeout=5.0):
                future.result()
        except TimeoutError:
            logger.warning("🦾⚠️ Emergency stop timed out, continuing...")

    def _ensure_sleep_pose_via_reconnect(self):
        """
        Phase 2: ALWAYS ensure arms return to sleep pose.
        Reconnects if needed to guarantee this critical safety requirement.
        """
        logger.debug("🤖 Phase 2: Ensuring sleep pose via reconnect")
        
        max_attempts = 3
        for attempt in range(max_attempts):
            logger.info(f"🦾 Sleep pose attempt {attempt + 1}/{max_attempts}")
            
            # Check if arms are responsive for sleep pose movement
            arms_responsive = self._check_arm_responsiveness()
            
            if not arms_responsive and attempt < max_attempts - 1:
                logger.info(f"🦾🔄 Arms not responsive (attempt {attempt + 1}); reconnecting...")
                try:
                    # Clear current connections
                    self._force_cleanup_connections()
                    
                    # Reconnect with error clearing
                    self._connect_arms(clear_error=True)
                    
                    # Allow time for connection stabilization
                    time.sleep(self.config.reboot_wait_time)
                    
                    # Re-check responsiveness
                    arms_responsive = self._check_arm_responsiveness()
                    
                except Exception as e:
                    logger.warning(f"🦾⚠️ Reconnection attempt {attempt + 1} failed: {e}")
                    continue
            
            # Attempt to move to sleep pose
            if arms_responsive:
                try:
                    logger.info("🦾 Moving arms to sleep pose")
                    
                    # CRITICAL: Switch from idle back to position mode before movement
                    for arm in (self.arm_l, self.arm_r):
                        if arm is not None:
                            arm.set_all_modes(trossen_arm.Mode.position)
                    
                    self.send_action(
                        self._urdf_joints_to_action(self.home_pos_full),
                        goal_time=self.config.goal_time * 2.0,  # Extra time for reliability
                        safe=True  # Block and verify positions
                    )
                    logger.info("✅🦾 Arms successfully moved to sleep pose")
                    return  # Success - exit the retry loop
                    
                except Exception as e:
                    logger.warning(f"🦾⚠️ Sleep pose movement failed (attempt {attempt + 1}): {e}")
                    if attempt < max_attempts - 1:
                        continue  # Try again
            else:
                logger.warning(f"🦾⚠️ Arms still not responsive after reconnect (attempt {attempt + 1})")
        
        # If we get here, all attempts failed
        logger.error("🦾❌ Failed to ensure sleep pose after all attempts")
        logger.error("🦾❌ SAFETY WARNING: Arms may not be in safe position")

    def _force_cleanup_connections(self):
        """Force cleanup of existing arm connections."""
        logger.debug("🦾 Force cleaning up existing connections")
        
        def force_cleanup_arm(arm, label):
            if arm is not None:
                try:
                    # Try normal cleanup first
                    arm.cleanup()
                except Exception:
                    try:
                        # Try cleanup with reboot
                        arm.cleanup(reboot_controller=True)
                        # Wait for reboot to complete
                        time.sleep(self.config.reboot_wait_time)
                    except Exception as e:
                        logger.warning(f"🦾⚠️ Force cleanup failed for {label} arm: {e}")

        # Force cleanup both arms
        futures = []
        if self.arm_l is not None:
            futures.append(self._executor.submit(force_cleanup_arm, self.arm_l, "Left"))
        if self.arm_r is not None:
            futures.append(self._executor.submit(force_cleanup_arm, self.arm_r, "Right"))
        
        # Wait for cleanups with timeout
        try:
            for future in as_completed(futures, timeout=10.0):
                future.result()
        except TimeoutError:
            logger.warning("🦾⚠️ Force cleanup timed out")
        
        # Clear references
        self.arm_l = None
        self.arm_r = None

    def _check_arm_responsiveness(self) -> bool:
        """Check if arms are responsive to commands."""
        try:
            # Try to get positions from connected arms
            responsive_count = 0
            total_arms = 0
            
            if self.arm_l is not None:
                total_arms += 1
                try:
                    self._get_positions_l()
                    responsive_count += 1
                except Exception:
                    pass
                    
            if self.arm_r is not None:
                total_arms += 1
                try:
                    self._get_positions_r()
                    responsive_count += 1
                except Exception:
                    pass
            
            # Both connected arms must be responsive
            is_responsive = total_arms > 0 and responsive_count == total_arms
            logger.debug(f"🦾 Arm responsiveness: {responsive_count}/{total_arms} arms responsive")
            return is_responsive
            
        except Exception as e:
            logger.debug(f"🦾 Arms not responsive: {e}")
            return False

    def _final_cleanup_and_shutdown(self, reboot_controller: bool = False):
        """Phase 3: Final cleanup and system shutdown."""
        logger.debug("🤖 Phase 3: Final cleanup and shutdown")
        
        # Set arms to idle mode one final time
        self._set_final_idle_mode()
        
        # Clean up arm connections
        self._cleanup_arm_connections(reboot_controller)
        
        # Shutdown system components
        self._shutdown_system()

    def _set_final_idle_mode(self):
        """Set arms to idle mode for final shutdown."""
        logger.debug("🦾 Setting arms to final idle mode")
        
        def set_arm_idle(arm, label):
            if arm is not None:
                try:
                    arm.set_all_modes(trossen_arm.Mode.idle)
                    logger.info(f"✅🦾 {label} arm set to final idle")
                except Exception as e:
                    logger.warning(f"🦾⚠️ Failed to set {label} arm to final idle: {e}")

        # Set both arms to idle
        futures = []
        if self.arm_l is not None:
            futures.append(self._executor.submit(set_arm_idle, self.arm_l, "Left"))
        if self.arm_r is not None:
            futures.append(self._executor.submit(set_arm_idle, self.arm_r, "Right"))
        
        # Wait for completion with timeout
        try:
            for future in as_completed(futures, timeout=5.0):
                future.result()
        except TimeoutError:
            logger.warning("🦾⚠️ Final idle mode setting timed out")

    def _cleanup_arm_connections(self, reboot_controller: bool = False):
        """Clean up arm driver connections with error reporting."""
        logger.debug("🦾 Cleaning up arm connections")
        
        def cleanup_arm(arm, label):
            if arm is not None:
                try:
                    # Log any final errors
                    error_info = arm.get_error_information()
                    if error_info and error_info.strip():
                        logger.info(f"🦾 {label} arm final status: {error_info}")
                    
                    # Normal cleanup
                    arm.cleanup(reboot_controller=reboot_controller)
                    logger.info(f"✅🦾 {label} arm connection cleaned up")
                    
                except Exception as e:
                    logger.warning(f"🦾⚠️ Normal cleanup failed for {label} arm: {e}")

        # Cleanup both arms
        futures = []
        if self.arm_l is not None:
            futures.append(self._executor.submit(cleanup_arm, self.arm_l, "Left"))
        if self.arm_r is not None:
            futures.append(self._executor.submit(cleanup_arm, self.arm_r, "Right"))
        
        # Wait for all cleanups
        try:
            for future in as_completed(futures, timeout=10.0):
                future.result()
        except TimeoutError:
            logger.warning("🦾⚠️ Cleanup timed out, forcing shutdown")
        
        # Clear references
        self.arm_l = None
        self.arm_r = None

    def _shutdown_system(self):
        """Shutdown cameras and thread pool."""
        logger.debug("🤖 System shutdown")
        
        # Disconnect cameras with retries
        for attempt in range(2):
            try:
                self._disconnect_cameras()
                logger.info("✅🎥 Cameras disconnected")
                break
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"🎥⚠️ Camera disconnect failed, retrying: {e}")
                    time.sleep(1.0)
                else:
                    logger.error(f"🎥❌ Camera disconnect failed after retries: {e}")
        
        # Graceful thread pool shutdown with Python 3.8 compatibility
        try:
            logger.debug("🧵 Shutting down thread pool gracefully")
            try:
                # Python 3.9+ supports cancel_futures
                self._executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                # Fallback for Python 3.8
                self._executor.shutdown(wait=True)
            logger.info("✅🧵 Thread pool shut down gracefully")
        except Exception as e:
            logger.warning(f"🧵⚠️ Graceful shutdown failed: {e}")
            try:
                try:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    self._executor.shutdown(wait=False)
                logger.info("🧵 Thread pool force shutdown completed")
            except Exception as force_e:
                logger.error(f"🧵❌ Force shutdown failed: {force_e}")
        finally:
            # Clear executor reference to enable hot reconnection
            self._executor = None

    def _emergency_cleanup(self):
        """Emergency cleanup when all else fails."""
        logger.warning("🤖🚨 Performing emergency cleanup")
        
        try:
            # Force stop any remaining motion
            if self.arm_l is not None:
                try:
                    self.arm_l.set_all_modes(trossen_arm.Mode.idle)
                except Exception:
                    pass
            if self.arm_r is not None:
                try:
                    self.arm_r.set_all_modes(trossen_arm.Mode.idle)
                except Exception:
                    pass
            
            # Force cleanup with reboot
            self._force_cleanup_connections()
            
            # Force shutdown cameras and thread pool
            try:
                self._disconnect_cameras()
            except Exception:
                pass
            
            try:
                try:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    self._executor.shutdown(wait=False)
                self._executor = None
            except Exception:
                pass
                
            logger.warning("🤖🚨 Emergency cleanup completed")
            
        except Exception as e:
            logger.error(f"🤖❌ Emergency cleanup failed: {e}")
