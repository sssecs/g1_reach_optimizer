#!/usr/bin/env python3
import argparse, numpy as np
ap=argparse.ArgumentParser()
ap.add_argument("path",nargs="?",default="pose_bank_output/pose_bank.npz")
a=ap.parse_args()
d=np.load(a.path,allow_pickle=True)
for i in range(len(d["target_heights"])):
    print(
        f"[{i}] target={d['target_heights'][i]:.3f} "
        f"mode={d['modes'][i]} "
        + (f"state={'SELECTED' if d['selected'][i] else 'ACCEPTED' if d['accepted'][i] else 'REJECTED'} "
           if 'accepted' in d else "") +
        f"{'wrist_h' if 'left_forward' in d else 'hand_h'}=({d['left_height'][i]:.3f},{d['right_height'][i]:.3f}) "
        + (f"forward=({d['left_forward'][i]:.3f},{d['right_forward'][i]:.3f}) "
           f"wrist_clear={d['min_wrist_body_clearance'][i]:.3f} "
           f"forward_manip={d['forward_manipulability'][i]:.4f} "
           if 'left_forward' in d else "") +
        (f"shoulder_diff={abs(d['shoulder_height_difference'][i]):.3f} "
         if 'shoulder_height_difference' in d else "") +
        f"tau_ratio={d['max_torque_ratio'][i]:.3f} "
        f"clearance={d['min_clearance'][i]:.3f} "
        f"manip={d['manipulability'][i]:.4f} "
        f"ipopt={d['ipopt_status'][i]} "
        f"eq={d['max_eq_violation'][i]:.2e} "
        f"ineq={d['min_ineq_margin'][i]:.2e} "
        + (f"pitch=({d['pelvis_pitch_deg'][i]:.1f},{d['torso_pitch_deg'][i]:.1f})deg"
           if 'pelvis_pitch_deg' in d and 'torso_pitch_deg' in d else "")
    )
