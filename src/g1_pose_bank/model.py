from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import mujoco

from .config import Config

def _match_first(mapping: dict[str, float], name: str, default: float) -> float:
    for pattern, value in mapping.items():
        if re.fullmatch(pattern, name):
            return float(value)
    return float(default)

class G1Model:
    def __init__(self, cfg: Config):
        rcfg = cfg.section("robot")
        path = Path(rcfg["mjcf"]).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"MJCF not found: {path}\n"
                "Run scripts/vendor_mjlab_g1.py first."
            )
        self.path = path
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)

        self.joint_names, self.hinge_ids = [], []
        for jid in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE and name:
                self.joint_names.append(name)
                self.hinge_ids.append(jid)

        self.qpos_indices = [int(self.model.jnt_qposadr[j]) for j in self.hinge_ids]
        self.dof_indices = [int(self.model.jnt_dofadr[j]) for j in self.hinge_ids]
        self.lower = np.array([self.model.jnt_range[j,0] for j in self.hinge_ids], float)
        self.upper = np.array([self.model.jnt_range[j,1] for j in self.hinge_ids], float)

        effort_cfg = cfg.raw.get("effort_limits", {})
        self.effort = np.array(
            [_match_first(effort_cfg, n, 1e6) for n in self.joint_names], float
        )

        self.target_points = {}
        for side in ("left", "right"):
            key = f"{side}_target_point"
            self.target_points[side] = (
                rcfg[key] if key in rcfg else {"site": rcfg[f"{side}_target_site"]}
            )

        self.q_home = self._build_home(cfg)
        self.set_qpos(self.q_home)

        if bool(rcfg.get("auto_ground_initial", True)):
            feet = [c for c in cfg.raw["contact_points"].values()
                    if c.get("role") == "foot"]
            if feet:
                z0 = float(cfg.section("ground").get("z", 0.0))
                minz = min(self.contact_point(c)[0][2] for c in feet)
                self.q_home[2] += z0 - minz
                self.set_qpos(self.q_home)

    def site_id(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if i < 0:
            raise ValueError(f"Site not found: {name}")
        return i

    def body_id(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise ValueError(f"Body not found: {name}")
        return i

    def geom_id(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if i < 0:
            raise ValueError(f"Geom not found: {name}")
        return i

    def _build_home(self, cfg: Config) -> np.ndarray:
        rcfg = cfg.section("robot")
        q = np.zeros(self.model.nq)
        q[:3] = np.asarray(rcfg["initial_root_pos"], float)
        q[3:7] = np.asarray(rcfg["initial_root_quat_wxyz"], float)
        mapping = rcfg.get("initial_joint_regex_values", {})
        for name, qi in zip(self.joint_names, self.qpos_indices):
            v = 0.0
            for pattern, value in mapping.items():
                if re.fullmatch(pattern, name):
                    v = float(value)
            q[qi] = v
        return q

    def set_qpos(self, q: np.ndarray):
        self.data.qpos[:] = q
        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def body_pitch(self, name: str) -> float:
        """Body forward pitch in radians; positive means leaning forward.

        Measured against world vertical in the body's sagittal plane, so yaw
        does not change the value.
        """
        R = self.data.xmat[self.body_id(name)].reshape(3, 3)
        return float(np.arctan2(-R[2, 0], R[2, 2]))

    def target_pos(self, side: str) -> np.ndarray:
        return self.contact_point(self.target_points[side])[0]

    def target_jac(self, side: str) -> np.ndarray:
        return self.contact_point(self.target_points[side])[1]

    def site_pos(self, sid: int) -> np.ndarray:
        return self.data.site_xpos[sid].copy()

    def site_jac(self, sid: int) -> np.ndarray:
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jp, jr, sid)
        return jp

    def point_world(self, body: str, offset) -> np.ndarray:
        bid = self.body_id(body)
        R = self.data.xmat[bid].reshape(3,3)
        return self.data.xpos[bid] + R @ np.asarray(offset, float)

    def point_jac(self, body: str, offset) -> np.ndarray:
        bid = self.body_id(body)
        p = self.point_world(body, offset)
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jp, jr, p, bid)
        return jp

    def contact_point(self, spec: dict):
        if "site" in spec:
            sid = self.site_id(spec["site"])
            return self.site_pos(sid), self.site_jac(sid)
        return self.point_world(spec["body"], spec["offset"]), self.point_jac(
            spec["body"], spec["offset"]
        )

    def bias_force(self) -> np.ndarray:
        return self.data.qfrc_bias.copy()

    def geom_distance(self, a: str, b: str, distmax: float = 1.0) -> float:
        fromto = np.zeros(6)
        return float(mujoco.mj_geomDistance(
            self.model, self.data, self.geom_id(a), self.geom_id(b), distmax, fromto
        ))

    def contact_nullspace(self, active_specs: list[dict]) -> np.ndarray:
        if not active_specs:
            return np.eye(self.model.nv)
        Jc = np.vstack([self.contact_point(c)[1] for c in active_specs])
        _, S, Vt = np.linalg.svd(Jc, full_matrices=True)
        rank = int(np.sum(S > 1e-7))
        return Vt[rank:].T

    def constrained_forward_manipulability(
        self, active_specs: list[dict], heading_body: str, mode: str = "both",
        nullspace: np.ndarray | None = None,
    ) -> float:
        """Local wrist motion along the horizontal body heading with contacts fixed."""
        N = self.contact_nullspace(active_specs) if nullspace is None else nullspace
        if N.shape[1] == 0:
            return 0.0
        R = self.data.xmat[self.body_id(heading_body)].reshape(3, 3)
        forward_xy = R[:2, 0]
        forward_xy = forward_xy / max(np.linalg.norm(forward_xy), 1e-8)
        direction = np.array([forward_xy[0], forward_xy[1], 0.0])
        sides = ("left", "right") if mode == "both" else (mode,)
        values = [
            float(np.linalg.norm(direction @ self.target_jac(side)[:3] @ N))
            for side in sides
        ]
        return float(np.mean(values))

    def constrained_manipulability(
        self, active_specs: list[dict], mode: str = "both",
        nullspace: np.ndarray | None = None,
    ) -> float:
        N = self.contact_nullspace(active_specs) if nullspace is None else nullspace
        vals = []
        sides = []
        if mode in ("both", "left"):
            sides.append("left")
        if mode in ("both", "right"):
            sides.append("right")
        for side in sides:
            J = self.target_jac(side) @ N
            if J.size == 0:
                vals.append(0.0)
                continue
            s = np.linalg.svd(J, compute_uv=False)
            vals.append(float(s[-1]) if len(s) else 0.0)
        return float(np.mean(vals)) if vals else 0.0
