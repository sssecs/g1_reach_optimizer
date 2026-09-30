from __future__ import annotations

import argparse
import math

from .config import Config
from .bank import generate_bank


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/static_pose_bank.yaml")
    ap.add_argument(
        "--height",
        help="solve one wrist height in metres, or 'min' for the lowest configured height",
    )
    ap.add_argument(
        "--output-dir",
        help="output directory (default: pose_bank_output, or pose_bank_output_single with --height)",
    )
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    if args.height is not None:
        if args.height == "min":
            heights = cfg.section("task")["target_heights"]
            if not heights:
                ap.error("task.target_heights is empty")
            height = min(float(value) for value in heights)
        else:
            try:
                height = float(args.height)
            except ValueError:
                ap.error("--height must be a height in metres or 'min'")
        if not math.isfinite(height) or height <= 0:
            ap.error("--height must be a positive finite number")
        cfg.section("task")["target_heights"] = [height]

    output_dir = args.output_dir or (
        "pose_bank_output_single" if args.height is not None else "pose_bank_output"
    )
    print(f"Output directory: {output_dir}", flush=True)
    winners = generate_bank(cfg, output_dir)
    print("\nGenerated pose bank:")
    for i, c in enumerate(winners):
        print(
            f"[{i}] target={c.target_height:.3f} mode={c.mode:16s} "
            f"wrist_h=({c.left_height:.3f},{c.right_height:.3f}) "
            f"forward=({c.left_forward:.3f},{c.right_forward:.3f}) "
            f"shoulder_diff={abs(c.shoulder_height_difference):.3f} "
            f"wrist_clear={c.min_wrist_body_clearance:.3f} "
            f"tau_ratio={c.max_torque_ratio:.3f} "
            f"clearance={c.min_clearance:.3f}"
        )


if __name__ == "__main__":
    main()
