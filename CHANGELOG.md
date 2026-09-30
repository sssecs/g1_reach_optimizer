# 0.5.1

- Kneeling modes now allow heel lift by using forefoot support instead of forcing
  all simplified sole points to remain on the ground.
- Added post-hoc physical feasibility checks independent of IPOPT status code.
- Added `ipopt_status`, `max_eq_violation`, and `min_ineq_margin` to console,
  JSON, NPZ and summary output.
- Added configurable `postcheck_eq_tol` and `postcheck_ineq_tol`.

