#!/usr/bin/env python3
"""Browse selected static poses in the MuJoCo viewer without simulating them."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
from mujoco.glfw import glfw


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_poses(path: Path, nq: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Pose bank not found: {path}")
    with np.load(path, allow_pickle=True) as bank:
        required = (
            "qpos", "target_heights", "modes", "left_height", "right_height",
            "max_torque_ratio", "min_clearance", "manipulability",
        )
        missing = [name for name in required if name not in bank]
        if missing:
            raise ValueError(f"Pose bank is missing: {', '.join(missing)}")
        poses = {name: bank[name].copy() for name in required}
        for name in (
            "pelvis_pitch_deg", "torso_pitch_deg", "left_forward", "right_forward",
            "min_wrist_body_clearance", "forward_manipulability",
            "shoulder_height_difference", "success", "accepted", "selected",
            "ipopt_status", "max_eq_violation", "min_ineq_margin",
        ):
            if name in bank:
                poses[name] = bank[name].copy()

    qpos = poses["qpos"]
    if qpos.ndim != 2 or qpos.shape[1] != nq or qpos.shape[0] == 0:
        raise ValueError(f"Expected nonempty qpos array with shape (N, {nq}); got {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise ValueError("Pose bank contains non-finite qpos values")
    for name, values in poses.items():
        if name != "qpos" and (values.ndim != 1 or len(values) != len(qpos)):
            raise ValueError(f"{name} must have one value per pose")
    return poses


def describe(poses: dict[str, np.ndarray], index: int) -> str:
    state = ""
    if "accepted" in poses:
        label = (
            "SELECTED" if poses["selected"][index] else
            "ACCEPTED" if poses["accepted"][index] else "REJECTED"
        )
        state = f"{label}  "
    diagnostics = ""
    if "max_eq_violation" in poses:
        diagnostics = (
            f"  ipopt={poses['ipopt_status'][index]}"
            f"  eq={poses['max_eq_violation'][index]:.2e}"
            f"  ineq={poses['min_ineq_margin'][index]:.2e}"
        )
    pitch = ""
    if "pelvis_pitch_deg" in poses and "torso_pitch_deg" in poses:
        pitch = (
            f"  pitch(pelvis,torso)=("
            f"{poses['pelvis_pitch_deg'][index]:.1f},"
            f"{poses['torso_pitch_deg'][index]:.1f}) deg"
        )
    forward = ""
    if "left_forward" in poses:
        forward = (
            f"  forward=({poses['left_forward'][index]:.3f},"
            f"{poses['right_forward'][index]:.3f}) m"
            f"  wrist_clear={poses['min_wrist_body_clearance'][index]:.3f} m"
            f"  forward_manip={poses['forward_manipulability'][index]:.4f}"
        )
    shoulder = ""
    if "shoulder_height_difference" in poses:
        shoulder = (
            f"  shoulder_diff="
            f"{abs(poses['shoulder_height_difference'][index]):.3f} m"
        )
    target_label = "wrists" if "left_forward" in poses else "hands"
    return (
        f"[{index + 1}/{len(poses['qpos'])}] {state}"
        f"target={poses['target_heights'][index]:.3f} m  "
        f"mode={poses['modes'][index]}  "
        f"{target_label}=({poses['left_height'][index]:.3f}, "
        f"{poses['right_height'][index]:.3f}) m  "
        f"torque_ratio={poses['max_torque_ratio'][index]:.3f}  "
        f"clearance={poses['min_clearance'][index]:.3f} m  "
        f"manipulability={poses['manipulability'][index]:.4f}"
        f"{forward}{shoulder}{pitch}{diagnostics}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bank", nargs="?", type=Path,
        default=PROJECT_ROOT / "pose_bank_output/pose_bank.npz",
        help="selected pose bank NPZ (default: pose_bank_output/pose_bank.npz)",
    )
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "assets/g1/scene_g1.xml",
        help="MuJoCo scene XML (default: assets/g1/scene_g1.xml)",
    )
    parser.add_argument("--index", type=int, default=0, help="initial pose index, starting at 0")
    parser.add_argument(
        "--interval", type=float, default=0,
        help="seconds between poses in automatic mode; 0 starts in manual mode",
    )
    args = parser.parse_args()

    if args.interval < 0:
        parser.error("--interval must be nonnegative")
    if not args.model.is_file():
        parser.error(f"MuJoCo model not found: {args.model}")
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    poses = load_poses(args.bank, model.nq)
    if "left_forward" not in poses:
        print("This bank predates the wrist-forward target; regenerate it before judging the new result.")
    count = len(poses["qpos"])
    if not 0 <= args.index < count:
        parser.error(f"--index must be between 0 and {count - 1}")

    data = mujoco.MjData(model)
    index = args.index
    autoplay = args.interval > 0
    interval = args.interval if autoplay else 2.0
    pending_keys: list[int] = []

    def on_key(key: int) -> None:
        if key in (glfw.KEY_LEFT, glfw.KEY_RIGHT, glfw.KEY_SPACE):
            pending_keys.append(key)

    print("Left/Right: previous/next pose  |  Space: toggle autoplay  |  Close window: quit")
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        last_change = time.monotonic()
        changed = True
        while viewer.is_running():
            while pending_keys:
                key = pending_keys.pop(0)
                if key == glfw.KEY_SPACE:
                    autoplay = not autoplay
                    print(f"Autoplay {'on' if autoplay else 'off'} ({interval:g} s)")
                else:
                    index = (index + (1 if key == glfw.KEY_RIGHT else -1)) % count
                    changed = True
                last_change = time.monotonic()

            if autoplay and time.monotonic() - last_change >= interval:
                index = (index + 1) % count
                changed = True
                last_change = time.monotonic()

            if changed:
                with viewer.lock():
                    data.qpos[:] = poses["qpos"][index]
                    data.qvel[:] = 0
                    mujoco.mj_forward(model, data)
                print(describe(poses, index), flush=True)
                changed = False

            viewer.sync()
            time.sleep(1 / 30)


if __name__ == "__main__":
    main()
