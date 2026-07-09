#!/usr/bin/env python
"""Minimal SpaceMouse end-effector teleop for a single YAM arm.

Reads one SpaceMouse and drives the arm's TCP pose through raiden's
manipulability-aware IK (PyRoki + J-PARSE) via ``EEPoseController``. No ``rd``
interface, no leader-follower, no recording — just EE pose control.

Axis mapping (world / base frame, tuned in raiden):
    push/pull puck  -> EE +x / -x       (SpaceMouse y)
    left/right puck -> EE -y / +y       (SpaceMouse x)
    up/down puck    -> EE +z / -z       (SpaceMouse z)
    tilt/twist      -> world-frame roll / pitch / yaw
    button 0 (held) -> close gripper
    button 1 (held) -> open gripper

Find your device's hidraw path with:
    uv run scripts/spacemouse_ee_control.py --list

Usage:
    uv run scripts/spacemouse_ee_control.py --channel can_follower_r
    uv run scripts/spacemouse_ee_control.py --channel can_follower_r --path /dev/hidraw7
"""

import argparse
import time

import numpy as np
import pyspacemouse
from scipy.spatial.transform import Rotation

from raiden.ee_control import EEPoseController


def find_spacemouse_paths() -> list[tuple[str, int, int, str]]:
    """Return (path, vid, pid, name) for each connected, supported SpaceMouse.

    pyspacemouse has no path lister, so we enumerate HID devices with its own
    ``easyhid`` backend and keep the ones whose (vid, pid) match a supported
    SpaceMouse spec. One puck exposes several HID interfaces that share a single
    hidraw path, so we dedupe by path.
    """
    import easyhid

    supported = {spec.hid_id for spec in pyspacemouse.get_device_specs().values()}
    seen: dict[str, tuple[int, int, str]] = {}
    for d in easyhid.Enumeration().find():
        if (d.vendor_id, d.product_id) not in supported:
            continue
        path = d.path.decode() if isinstance(d.path, bytes) else d.path
        seen.setdefault(path, (d.vendor_id, d.product_id, d.product_string or ""))
    return [(p, vid, pid, name) for p, (vid, pid, name) in seen.items()]


def spacemouse_to_target_pose(
    state,
    T_current: np.ndarray,
    vel_scale: float,
    rot_scale: float,
    invert_rotation: bool = False,
) -> np.ndarray:
    """Convert a SpaceMouse state into a target TCP pose (world-frame deltas)."""
    rot_sign = -1.0 if invert_rotation else 1.0

    dx = state.y * vel_scale
    dy = -state.x * vel_scale
    dz = state.z * vel_scale
    drx = rot_sign * state.roll * rot_scale
    dry = rot_sign * state.pitch * rot_scale
    drz = -rot_sign * state.yaw * rot_scale

    T_target = T_current.copy()
    T_target[:3, 3] += [dx, dy, dz]
    T_target[:3, :3] = (
        Rotation.from_euler("xyz", [drx, dry, drz]).as_matrix() @ T_current[:3, :3]
    )
    return T_target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List connected SpaceMouse hidraw paths and exit.",
    )
    parser.add_argument(
        "--channel",
        default="can_follower_r",
        help="CAN channel for the YAM arm (default: can_follower_r).",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="hidraw path of the SpaceMouse (default: auto-detect first found).",
    )
    parser.add_argument(
        "--vel-scale",
        type=float,
        default=0.07,
        help="Max translational speed (m/s) at full puck deflection.",
    )
    parser.add_argument(
        "--rot-scale",
        type=float,
        default=0.8,
        help="Max rotational speed (rad/s) at full puck deflection.",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Control period (s).")
    parser.add_argument(
        "--invert-rotation", action="store_true", help="Negate all rotation axes."
    )
    parser.add_argument(
        "--no-home",
        action="store_true",
        help="Skip the move-to-home step at startup.",
    )
    args = parser.parse_args()

    if args.list:
        devices = find_spacemouse_paths()
        if not devices:
            print("No SpaceMouse devices found.")
        else:
            print("Connected SpaceMouse devices:")
            for path, vid, pid, name in devices:
                print(f"  {path}  {name} (VID={vid:#06x}, PID={pid:#06x})")
        return

    path = args.path
    if path is None:
        devices = find_spacemouse_paths()
        if not devices:
            raise SystemExit(
                "No SpaceMouse found. Plug in the puck, or run with --list to "
                "see detected devices and pass one to --path."
            )
        path = devices[0][0]
        print(f"Auto-detected SpaceMouse at {path} ({devices[0][3]}).")

    # Gripper command speed: full open/close over ~1 s of holding the button.
    gripper_step = args.dt / 1.0

    arm = EEPoseController(channel=args.channel, dt=args.dt)
    try:
        if not args.no_home:
            print("Moving arm to home position...")
            arm.move_to_home()

        print(f"Opening SpaceMouse ({path})...")
        mouse = pyspacemouse.open_by_path(path)

        print(
            "\n" + "=" * 60 + "\n"
            "  SPACEMOUSE EE CONTROL ACTIVE\n" + "=" * 60 + "\n"
            "  Move/tilt puck to move EE. Button0=close, Button1=open.\n"
            "  Press Ctrl+C to stop.\n" + "=" * 60 + "\n"
        )

        gripper = arm.get_gripper()
        while True:
            loop_start = time.monotonic()

            state = mouse.read()
            if state is None:
                time.sleep(args.dt)
                continue

            T_current = arm.get_ee_pose()
            T_target = spacemouse_to_target_pose(
                state,
                T_current,
                args.vel_scale,
                args.rot_scale,
                args.invert_rotation,
            )

            # Gripper: button 0 closes, button 1 opens (while held).
            buttons = getattr(state, "buttons", [])
            if len(buttons) > 0 and buttons[0]:
                gripper = max(0.0, gripper - gripper_step)
            elif len(buttons) > 1 and buttons[1]:
                gripper = min(1.0, gripper + gripper_step)

            arm.servo_to_pose(T_target, gripper=gripper)

            remaining = args.dt - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C).")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
