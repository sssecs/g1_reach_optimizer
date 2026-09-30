# G1 Static Pose Bank Generator v0.5.1

This package generates a small set of quasi-static G1 key poses instead of one long
contact-transition trajectory. Dynamic transitions are left to RL.

For each requested wrist height it:
1. enumerates fixed support modes,
2. solves one static NLP per mode,
3. rejects infeasible candidates,
4. scores torque effort, joint margin, hand/leg clearance, contact-constrained
   manipulability and continuity from the previously selected pose,
5. keeps the best pose.

There is no contact complementarity.

Allowed support: feet, knees, thighs.
Hands, wrists, forearms and elbows are manipulation-only and never support the robot.

## Install

```bash
conda env create -f environment.yml
conda activate g1-pose-bank
python scripts/check_environment.py
```

## Copy the exact Unitree MJLab G1 model into the project

```bash
python scripts/vendor_mjlab_g1.py /path/to/unitree_rl_mjlab
```

This creates:

```text
assets/g1/
├── g1.xml
└── assets/
    └── *.STL
```

## Run

```bash
g1-pose-bank \
  --config configs/static_pose_bank.yaml \
  --output-dir pose_bank_output
```

Summarize:

```bash
python scripts/summarize_pose_bank.py pose_bank_output/pose_bank.npz
```

Outputs:
- `pose_bank_output/pose_bank.npz` (selected feasible poses)
- `pose_bank_output/all_candidates.npz` (every returned mode candidate)
- `pose_bank_output/candidates/height_*/` (one NPZ per mode)
- `pose_bank_output/pose_bank.json` (metrics and solver errors)

The JSON stores both selected poses and all attempted contact-mode candidates.

## Default targets

```yaml
target_heights: [0.55, 0.40, 0.25, 0.12]
```

Default modes:
- feet
- feet + left knee
- feet + right knee
- feet + both knees
- feet + both thighs

## Manipulation-space term

For active contact Jacobian `Jc`, the optimizer computes its nullspace `N` and evaluates

`J_wrist_free = J_wrist N`.

The smallest singular value of this projected translational Jacobian is rewarded. This
approximates how much local wrist motion remains after respecting the active support contacts.

## Intended RL use

The poses are key states, not a timed trajectory. Use them as:
- goal-conditioned targets,
- curriculum stages,
- reset distributions,
- reference states for transition learning.

RL handles inertia, impact, contact switching and balance recovery.


## v0.5.1 changes

Kneeling modes no longer force all foot support points to remain on the ground.

- `feet`: keeps the full simplified foot support polygon.
- `feet_left_knee`: forefoot/toe support + left knee.
- `feet_right_knee`: forefoot/toe support + right knee.
- `feet_both_knees`: forefoot/toe support + both knees.
- `feet_both_thighs`: forefoot/toe support + both thighs.

This permits heel lift and ankle pitch in kneeling/very-low configurations.

Candidate acceptance is also no longer based only on IPOPT's return status.
Every returned point is checked using:

```text
max_eq_violation <= postcheck_eq_tol
min_ineq_margin  >= -postcheck_ineq_tol
```

The default post-check tolerance is `1e-4`.

Console output and JSON/NPZ results now include:

- `ipopt_status`
- `max_eq_violation`
- `min_ineq_margin`

This makes it possible to distinguish "IPOPT stopped early" from "physically infeasible".

## Visualize the selected poses

Open the pose bank in the MuJoCo viewer (requires a desktop display):

```bash
python scripts/visualize_pose_bank.py pose_bank_output/pose_bank.npz
```

Use the left and right arrow keys to switch poses, or press Space to start/stop
automatic playback. The current pose's target height, contact mode, target heights,
torque ratio, clearance, and manipulability are printed in the terminal. Start
automatic playback immediately with `--interval 2`, or open a specific pose with
`--index 2`. The viewer shows static key poses; it does not simulate transitions.

## Backward lean limit

The default configuration limits both pelvis and torso backward pitch to 10 degrees
using `limits.minimum_body_pitch_deg` in `configs/static_pose_bank.yaml`.
Positive pitch means leaning forward. These are hard optimization constraints,
and candidates are saved only when the post-solve feasibility check passes.
The measured pitch angles are included in the JSON/NPZ outputs and shown by the
summary and visualization scripts. Regenerate the pose bank after changing this
limit; previously saved poses are not modified automatically.

## Wrist position and front/back reach

The target point is the center of each wrist collision capsule. `target_heights`
now specify wrist-center heights, so pose banks made with the old palm target
must be regenerated. The optimizer prefers each wrist 0.40 m horizontally
in front of the torso origin along the pelvis heading. This is a soft objective:
a low target may use a shorter reach when balance or clearance requires it.
Adjust `task.wrist_forward_preferred` and `weights.wrist_forward`.

`wrist_body_clearance_pairs` checks the wrist against the torso, pelvis, thighs,
and shins on both sides. The default surface clearance is at least 0.06 m,
with a 0.12 m preferred clearance in the objective. The optimizer also rewards
the wrist's contact-constrained Jacobian along the forward direction. This
rewards local front/back mobility while the selected pose itself satisfies
the static balance and collision constraints. It is a local mobility estimate;
it does not certify a collision-free, statically balanced trajectory between poses.

After editing these settings, run:

```bash
g1-pose-bank --config configs/static_pose_bank.yaml --output-dir pose_bank_output
python scripts/visualize_pose_bank.py pose_bank_output/pose_bank.npz
```

## Shoulder level and single-height validation

The left and right shoulder pitch-joint origins must differ in world height by
no more than 0.02 m. The optimizer also penalizes the remaining difference.
Change `limits.maximum_shoulder_height_difference` to adjust this tolerance.

To validate only the lowest configured wrist height, without overwriting the
existing full pose bank:

```bash
g1-pose-bank --config configs/static_pose_bank.yaml --height min
python scripts/summarize_pose_bank.py pose_bank_output_single/pose_bank.npz
python scripts/visualize_pose_bank.py pose_bank_output_single/pose_bank.npz
```

`--height 0.12` also selects a specific height in metres. A single-height solve
starts from the configured home pose; it does not use higher poses as warm starts.

## Review every contact mode

Each run saves all returned mode candidates, including rejected poses:

```text
pose_bank_output_single/all_candidates.npz
pose_bank_output_single/candidates/height_0p120/*.npz
pose_bank_output_single/pose_bank.json
```

After rerunning the lowest-height solve, browse all modes with:

```bash
python scripts/summarize_pose_bank.py pose_bank_output_single/all_candidates.npz
python scripts/visualize_pose_bank.py pose_bank_output_single/all_candidates.npz
```

The viewer's left/right keys step through candidates. `SELECTED`, `ACCEPTED`,
and `REJECTED` identify selection and feasibility; equality and inequality
violations are shown alongside each pose. Individual NPZ files under
`candidates/height_0p120/` can also be opened directly. A rejected pose is
provided for diagnosis and may violate balance, contact, or clearance limits.
