"""Minimal end-effector pose control for a single YAM arm.

This is a standalone, singularity-aware EE pose interface built on raiden's
manipulability-aware IK (PyRoki + J-PARSE). It deliberately does NOT depend on
``RobotController`` / the ``rd`` interface — it talks to one i2rt YAM arm
directly and exposes a tiny API:

    arm = EEPoseController(channel="can_follower_r")
    arm.move_to_home()
    T = arm.get_ee_pose()          # 4x4 TCP pose in the arm base frame
    T[:3, 3] += [0.01, 0, 0]       # nudge target +1 cm in x
    arm.servo_to_pose(T)           # one J-PARSE IK step + joint command
    arm.set_gripper(0.0)           # 0 = closed, 1 = open

The J-PARSE step is JIT-compiled once and warmed up in ``__init__`` so the
first real ``servo_to_pose`` call is not delayed by JAX tracing.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path as _Path
from typing import Optional

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import yourdfpy
from scipy.spatial.transform import Rotation

# Make the vendored i2rt importable (mirrors raiden.robot.controller).
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "third_party", "i2rt")
)

from i2rt.robots.get_robot import get_yam_robot  # noqa: E402
from i2rt.robots.robot import Robot  # noqa: E402
from i2rt.robots.utils import ARM_YAM_XML_PATH as _ARM_YAM_XML_PATH  # noqa: E402
from i2rt.robots.utils import GripperType  # noqa: E402

from raiden.robot._jparse import jparse_step  # noqa: E402

# ---------------------------------------------------------------------------
# Geometry / robot constants (kept identical to raiden.robot.controller so the
# IK behaves the same, but copied here to stay independent of RobotController).
# ---------------------------------------------------------------------------

_YAM_URDF_PATH = str(_Path(_ARM_YAM_XML_PATH).with_suffix(".urdf"))
_YAM_ASSETS_DIR = str(_Path(_ARM_YAM_XML_PATH).parent / "assets")

# Home configuration: 6 arm joints + gripper (gripper 1.0 = open).
HOME_POS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

# Follower PD gains (from raiden.robot.controller).
FOLLOWER_KP = np.array([80.0, 80.0, 80.0, 40.0, 10.0, 10.0, 20.0])
FOLLOWER_KD = np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5, 0.5])

# Fixed transform from link_6 origin to the tcp_site (grasp point).
_T_LINK6_TO_TCP: np.ndarray = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
# Inverse (pure rotation, so inverse = transpose).
_T_TCP_TO_LINK6: np.ndarray = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _load_yam_urdf() -> yourdfpy.URDF:
    """Load the YAM URDF, resolving package:// asset paths."""

    def _pkg_handler(fname, dir=None):  # noqa: A002
        if isinstance(fname, str) and fname.startswith("package://assets/"):
            return fname.replace("package://assets/", _YAM_ASSETS_DIR + "/")
        return fname

    return yourdfpy.URDF.load(
        _YAM_URDF_PATH,
        filename_handler=_pkg_handler,
        load_meshes=False,
        load_collision_meshes=True,
    )


class EEPoseController:
    """Singularity-aware EE pose control for one YAM arm (PyRoki + J-PARSE).

    Joint-order note: i2rt/MuJoCo and PyRoki/URDF use reversed arm-joint order,
    so we reverse at the two hardware boundaries (read / command). All internal
    state (``self._q_arm``) and FK/IK are in PyRoki order.
    """

    def __init__(
        self,
        channel: str = "can_follower_r",
        gripper_type: GripperType = GripperType.LINEAR_4310,
        dt: float = 0.01,
    ) -> None:
        """Connect to the arm and warm up the J-PARSE IK.

        Args:
            channel:      CAN channel for the YAM follower arm.
            gripper_type: i2rt gripper type.
            dt:           IK integration time step (s). Match your control loop.
        """
        self.dt = float(dt)

        print(f"Connecting to YAM arm on '{channel}'...")
        self.robot: Robot = get_yam_robot(
            channel=channel,
            gripper_type=gripper_type,
            zero_gravity_mode=False,
        )
        self.robot.update_kp_kd(kp=FOLLOWER_KP, kd=FOLLOWER_KD)

        # PyRoki model + JIT-compiled J-PARSE step.
        print("Loading PyRoki model and compiling J-PARSE IK (JIT warmup)...")
        urdf = _load_yam_urdf()
        self._pk_robot = pk.Robot.from_urdf(urdf)
        self._link6_idx = list(self._pk_robot.links.names).index("link_6")
        self._step_jit = jax.jit(jparse_step, static_argnames=("method",))
        self._home_cfg = np.zeros(6, dtype=np.float64)
        self._warmup()
        print("J-PARSE IK ready.")

        # Seed virtual arm state from the real robot (PyRoki order).
        q_full = self.robot.get_joint_pos()
        self._q_arm = q_full[:6][::-1].astype(np.float64).copy()  # MuJoCo -> PyRoki
        self._gripper = float(q_full[6])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _warmup(self) -> None:
        """Force JIT compilation so the first real IK call is not delayed."""
        dummy = np.zeros(6, dtype=np.float64)
        result, _ = self._step_jit(
            robot=self._pk_robot,
            cfg=dummy,
            target_link_index=self._link6_idx,
            target_position=np.zeros(3, dtype=np.float64),
            target_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            method="jparse",
            dt=self.dt,
            home_cfg=dummy,
        )
        jax.block_until_ready(result)

    def _fk_tcp(self, q_arm: np.ndarray) -> np.ndarray:
        """FK of the TCP (grasp site): link_6 pose x fixed offset -> 4x4."""
        poses = self._pk_robot.forward_kinematics(jnp.asarray(q_arm))
        T_link6 = np.array(jaxlie.SE3(poses[self._link6_idx]).as_matrix())
        return T_link6 @ _T_LINK6_TO_TCP

    def _ik_step(self, q_arm: np.ndarray, T_target_tcp: np.ndarray) -> np.ndarray:
        """One J-PARSE velocity-IK step toward a TCP target pose."""
        T_target_link6 = T_target_tcp @ _T_TCP_TO_LINK6
        target_pos = T_target_link6[:3, 3]
        xyzw = Rotation.from_matrix(T_target_link6[:3, :3]).as_quat()
        target_wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
        q_new, _ = self._step_jit(
            robot=self._pk_robot,
            cfg=q_arm.astype(np.float64),
            target_link_index=self._link6_idx,
            target_position=target_pos,
            target_wxyz=target_wxyz,
            method="jparse",
            dt=self.dt,
            home_cfg=self._home_cfg,
        )
        return np.asarray(q_new)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ee_pose(self) -> np.ndarray:
        """Return the current commanded TCP pose as a 4x4 matrix (base frame)."""
        return self._fk_tcp(self._q_arm)

    def get_gripper(self) -> float:
        """Return the last commanded gripper value (0 closed .. 1 open)."""
        return self._gripper

    def set_gripper(self, value: float) -> None:
        """Set the gripper command (0 = closed, 1 = open), clamped to [0, 1]."""
        self._gripper = float(np.clip(value, 0.0, 1.0))

    def servo_to_pose(
        self, T_target: np.ndarray, gripper: Optional[float] = None
    ) -> np.ndarray:
        """Take one J-PARSE IK step toward ``T_target`` and command the arm.

        ``T_target`` is a 4x4 TCP pose in the arm base frame. The step is
        velocity-limited and singularity-aware, so call this repeatedly in a
        control loop (it moves toward the target, not instantly to it).

        Args:
            T_target: 4x4 desired TCP pose in the arm base frame.
            gripper:  Optional gripper command (0..1). If None, holds the last.

        Returns:
            The full 7-DOF joint command sent to the robot (MuJoCo order).
        """
        self._q_arm = self._ik_step(self._q_arm, T_target)
        if gripper is not None:
            self.set_gripper(gripper)
        cmd = np.append(self._q_arm[::-1], self._gripper)  # PyRoki -> MuJoCo
        self.robot.command_joint_pos(cmd)
        return cmd

    def move_to_home(self, time_interval_s: float = 3.0, steps: int = 200) -> None:
        """Smoothly interpolate all joints to the home configuration."""
        start = self.robot.get_joint_pos().astype(np.float64)
        for i in range(steps + 1):
            alpha = i / steps
            self.robot.command_joint_pos((1 - alpha) * start + alpha * HOME_POS)
            if i < steps:
                time.sleep(time_interval_s / steps)
        # Re-sync virtual state to the new (home) pose.
        q_full = self.robot.get_joint_pos()
        self._q_arm = q_full[:6][::-1].astype(np.float64).copy()
        self._gripper = float(q_full[6])

    def resync_from_robot(self) -> None:
        """Reset internal virtual state to the robot's measured position.

        Call this after the arm has been moved by something other than
        ``servo_to_pose`` (e.g. an external hold) to avoid a jump.
        """
        q_full = self.robot.get_joint_pos()
        self._q_arm = q_full[:6][::-1].astype(np.float64).copy()
        self._gripper = float(q_full[6])

    def close(self) -> None:
        """Close the underlying robot connection."""
        self.robot.close()
