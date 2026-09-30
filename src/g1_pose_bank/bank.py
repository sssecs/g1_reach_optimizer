from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np

from .config import Config
from .static_optimizer import PoseCandidate, StaticPoseOptimizer


def _candidate_score(c: PoseCandidate, cfg: Config) -> float:
    if not c.success:
        return float("inf")
    tol = float(cfg.section("task").get("height_acceptance", 0.015))
    tm = cfg.section("task").get("target_mode", "both")
    errs = []
    if tm in ("both", "left"):
        errs.append(abs(c.left_height - c.target_height))
    if tm in ("both", "right"):
        errs.append(abs(c.right_height - c.target_height))
    if errs and max(errs) > tol:
        return float("inf")
    return c.objective


def _candidate_record(c: PoseCandidate, score: float) -> dict:
    return {
        "target_height": c.target_height,
        "mode": c.mode,
        "success": c.success,
        "status": c.status,
        "objective": c.objective,
        "left_height": c.left_height,
        "right_height": c.right_height,
        "left_forward": c.left_forward,
        "right_forward": c.right_forward,
        "min_wrist_body_clearance": c.min_wrist_body_clearance,
        "forward_manipulability": c.forward_manipulability,
        "shoulder_height_difference": c.shoulder_height_difference,
        "max_torque_ratio": c.max_torque_ratio,
        "min_clearance": c.min_clearance,
        "manipulability": c.manipulability,
        "body_pitch_deg": c.body_pitch_deg,
        "ipopt_status": c.ipopt_status,
        "max_eq_violation": c.max_eq_violation,
        "min_ineq_margin": c.min_ineq_margin,
        "selection_score": score if np.isfinite(score) else None,
        "selected": False,
    }


def _save_poses(
    path: Path, candidates: list[PoseCandidate], cfg: Config, selected_ids: set[int]
) -> None:
    """Save poses in the same layout for winners, all candidates and one-mode files."""
    scores = [_candidate_score(c, cfg) for c in candidates]
    np.savez_compressed(
        path,
        target_heights=np.array([c.target_height for c in candidates]),
        modes=np.array([c.mode for c in candidates], dtype=object),
        qpos=np.stack([c.qpos for c in candidates]),
        torque=np.stack([c.torque for c in candidates]),
        max_torque_ratio=np.array([c.max_torque_ratio for c in candidates]),
        min_clearance=np.array([c.min_clearance for c in candidates]),
        manipulability=np.array([c.manipulability for c in candidates]),
        pelvis_pitch_deg=np.array([
            c.body_pitch_deg.get("pelvis", np.nan) for c in candidates
        ]),
        torso_pitch_deg=np.array([
            c.body_pitch_deg.get("torso_link", np.nan) for c in candidates
        ]),
        ipopt_status=np.array([c.ipopt_status for c in candidates]),
        max_eq_violation=np.array([c.max_eq_violation for c in candidates]),
        min_ineq_margin=np.array([c.min_ineq_margin for c in candidates]),
        left_height=np.array([c.left_height for c in candidates]),
        right_height=np.array([c.right_height for c in candidates]),
        left_forward=np.array([c.left_forward for c in candidates]),
        right_forward=np.array([c.right_forward for c in candidates]),
        min_wrist_body_clearance=np.array([
            c.min_wrist_body_clearance for c in candidates
        ]),
        forward_manipulability=np.array([
            c.forward_manipulability for c in candidates
        ]),
        shoulder_height_difference=np.array([
            c.shoulder_height_difference for c in candidates
        ]),
        contact_names=np.array([
            c.active_contact_names for c in candidates
        ], dtype=object),
        contact_forces=np.array([c.forces for c in candidates], dtype=object),
        success=np.array([c.success for c in candidates], dtype=bool),
        accepted=np.array([np.isfinite(score) for score in scores], dtype=bool),
        selected=np.array([id(c) in selected_ids for c in candidates], dtype=bool),
        selection_score=np.array(scores),
    )


def generate_bank(cfg: Config, output_dir: str | Path):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    opt = StaticPoseOptimizer(cfg)
    heights = [float(x) for x in cfg.section("task")["target_heights"]]
    mode_order = list(cfg.raw["contact_modes"].keys())

    qref = opt.g1.q_home.copy()
    winners: list[PoseCandidate] = []
    all_candidates: list[PoseCandidate] = []
    all_rows: list[dict] = []
    candidate_rows: list[dict] = []

    for h in heights:
        print(f"\n===== target wrist height {h:.3f} m =====")
        candidates: list[PoseCandidate] = []
        rows_this_height: list[dict] = []
        for mode in mode_order:
            print(f"  solving mode: {mode}")
            try:
                c = opt.solve(mode, h, qref)
            except Exception as e:
                print("    ERROR:", repr(e))
                all_rows.append({
                    "target_height": h, "mode": mode, "success": False,
                    "selected": False, "error": repr(e),
                })
                continue

            candidates.append(c)
            all_candidates.append(c)
            score = _candidate_score(c, cfg)
            print(
                f"    success={c.success} objective={c.objective:.3f} "
                f"wrist_h=({c.left_height:.3f},{c.right_height:.3f}) "
                f"forward=({c.left_forward:.3f},{c.right_forward:.3f}) "
                f"wrist_clear={c.min_wrist_body_clearance:.3f} "
                f"shoulder_diff={abs(c.shoulder_height_difference):.3f} "
                f"tau_ratio={c.max_torque_ratio:.3f} "
                f"clearance={c.min_clearance:.3f} manip={c.manipulability:.4f} "
                f"ipopt={c.ipopt_status} "
                f"eq={c.max_eq_violation:.2e} ineq={c.min_ineq_margin:.2e}"
            )
            row = _candidate_record(c, score)
            all_rows.append(row)
            rows_this_height.append(row)
            candidate_rows.append(row)

        valid = [c for c in candidates if np.isfinite(_candidate_score(c, cfg))]
        if not valid:
            print("  WARNING: no accepted mode at this height.")
            continue
        best = min(valid, key=lambda c: _candidate_score(c, cfg))
        winners.append(best)
        qref = best.qpos.copy()
        best_index = next(i for i, candidate in enumerate(candidates) if candidate is best)
        rows_this_height[best_index]["selected"] = True
        print(f"  ==> selected: {best.mode}")

    selected_ids = {id(c) for c in winners}
    if all_candidates:
        _save_poses(out / "all_candidates.npz", all_candidates, cfg, selected_ids)
        for index, (candidate, row) in enumerate(zip(all_candidates, candidate_rows)):
            height_dir = out / "candidates" / f"height_{candidate.target_height:.3f}".replace(".", "p")
            height_dir.mkdir(parents=True, exist_ok=True)
            safe_mode = re.sub(r"[^a-zA-Z0-9_-]", "_", candidate.mode)
            path = height_dir / f"{index:02d}_{safe_mode}.npz"
            _save_poses(path, [candidate], cfg, selected_ids)
            row["candidate_file"] = str(path.relative_to(out))

    if winners:
        _save_poses(out / "pose_bank.npz", winners, cfg, selected_ids)
    with open(out / "pose_bank.json", "w", encoding="utf-8") as f:
        json.dump({
            "selected": [
                dict(_candidate_record(c, _candidate_score(c, cfg)),
                     selected=True, contacts=c.active_contact_names)
                for c in winners
            ],
            "all_candidates": all_rows,
        }, f, indent=2, ensure_ascii=False)

    if all_candidates:
        print(f"Saved {len(all_candidates)} mode candidates to {out / 'all_candidates.npz'}")
    else:
        print(f"No mode returned a pose; see {out / 'pose_bank.json'}")
    if not winners:
        raise RuntimeError(
            "No feasible static poses were generated. Inspect all_candidates.npz "
            "and pose_bank.json for rejected modes."
        )
    return winners
