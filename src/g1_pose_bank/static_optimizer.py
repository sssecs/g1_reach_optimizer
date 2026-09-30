from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import lsq_linear
import cyipopt

from .config import Config
from .model import G1Model

@dataclass
class PoseCandidate:
    target_height: float
    mode: str
    success: bool
    status: str
    objective: float
    qpos: np.ndarray
    forces: np.ndarray
    active_contact_names: list[str]
    left_height: float
    right_height: float
    left_forward: float
    right_forward: float
    min_wrist_body_clearance: float
    forward_manipulability: float
    shoulder_height_difference: float
    torque: np.ndarray
    max_torque_ratio: float
    min_clearance: float
    manipulability: float
    body_pitch_deg: dict[str, float]
    ipopt_status: int
    max_eq_violation: float
    min_ineq_margin: float

class StaticPoseOptimizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.g1 = G1Model(cfg)
        self.contacts = cfg.raw["contact_points"]
        self.modes = cfg.raw["contact_modes"]
        self.task = cfg.section("task")
        robot = cfg.section("robot")
        self.left_shoulder_id = self.g1.body_id(
            robot.get("left_shoulder_body", "left_shoulder_pitch_link")
        )
        self.right_shoulder_id = self.g1.body_id(
            robot.get("right_shoulder_body", "right_shoulder_pitch_link")
        )
        self.ground = cfg.section("ground")
        self.weights = cfg.section("weights")
        self.limits = cfg.section("limits")
        self.clearance_pairs = cfg.raw.get("clearance_pairs", [])
        self.wrist_clearance_pairs = cfg.raw.get("wrist_body_clearance_pairs", [])
        self.collision_pairs = cfg.raw.get("collision_pairs", [])
        self.solver = cfg.section("solver")

    def active_specs(self, mode: str):
        return [(n, self.contacts[n]) for n in self.modes[mode]]

    def pack(self, q, F):
        return np.concatenate([q, F.reshape(-1)])

    def unpack(self, x, mode):
        na = len(self.modes[mode])
        q = x[:self.g1.model.nq]
        F = x[self.g1.model.nq:].reshape(na, 3)
        return q, F

    def bounds(self, mode, qref):
        rad = np.asarray(self.limits["base_translation_radius"], float)
        b = []
        for i in range(3):
            b.append((qref[i]-rad[i], qref[i]+rad[i]))
        b += [(-1.0, 1.0)] * 4
        b += [(float(lo), float(hi)) for lo,hi in zip(self.g1.lower, self.g1.upper)]
        fmax = float(self.ground["max_normal_force"])
        for _ in self.modes[mode]:
            b += [(-fmax,fmax),(-fmax,fmax),(0.0,fmax)]
        return b

    def static_force_seed(self, q, mode):
        active = self.active_specs(mode)
        self.g1.set_qpos(q)
        bias = self.g1.bias_force()[:6]
        Acols = []
        for _,spec in active:
            _,J = self.g1.contact_point(spec)
            Acols.append(J.T[:6,2])
        A = np.stack(Acols, axis=1)
        fmax = float(self.ground["max_normal_force"])
        sol = lsq_linear(A, bias, bounds=(0.0, fmax), lsmr_tol="auto")
        F = np.zeros((len(active),3))
        F[:,2] = sol.x
        return F

    def initial_guess(self, mode, qref):
        return self.pack(qref.copy(), self.static_force_seed(qref, mode))

    def node(self, q, F, mode):
        self.g1.set_qpos(q)
        residual = self.g1.bias_force().copy()
        gaps = {}
        for (name,spec), f in zip(self.active_specs(mode),F):
            p,J = self.g1.contact_point(spec)
            gaps[name] = p[2]-float(self.ground["z"])
            residual -= J.T @ f
        tau = residual[self.g1.dof_indices]
        left = self.g1.target_pos("left")
        right = self.g1.target_pos("right")
        origin = self.g1.data.xpos[
            self.g1.body_id(self.task.get("wrist_forward_reference_body", "torso_link"))
        ]
        heading_body = self.task.get("wrist_forward_heading_body", "pelvis")
        R = self.g1.data.xmat[self.g1.body_id(heading_body)].reshape(3, 3)
        heading = R[:2, 0]
        heading = heading / max(np.linalg.norm(heading), 1e-8)
        return {
            "residual": residual,
            "tau": tau,
            "hL": left[2],
            "hR": right[2],
            "forwardL": float((left[:2] - origin[:2]) @ heading),
            "forwardR": float((right[:2] - origin[:2]) @ heading),
            "shoulder_height_difference": float(
                self.g1.data.xpos[self.left_shoulder_id, 2]
                - self.g1.data.xpos[self.right_shoulder_id, 2]
            ),
            "gaps": gaps,
        }

    def all_gaps(self):
        z0=float(self.ground["z"])
        return {n:self.g1.contact_point(s)[0][2]-z0 for n,s in self.contacts.items()}

    def min_wrist_body_clearance(self):
        if not self.wrist_clearance_pairs:
            return 1.0
        return min(
            self.g1.geom_distance(a, b, 0.5)
            for a, b in self.wrist_clearance_pairs
        )

    def min_clearance(self):
        if not self.clearance_pairs:
            return 1.0
        return min(self.g1.geom_distance(a,b,0.5) for a,b in self.clearance_pairs)

    def objective(self, x, mode, target, qref):
        q,F=self.unpack(x,mode)
        n=self.node(q,F,mode)
        w=self.weights
        tm=self.task.get("target_mode","both")

        err=0.0
        if tm in ("both","left"): err += (n["hL"]-target)**2
        if tm in ("both","right"): err += (n["hR"]-target)**2
        c=float(w["hand_height"])*err

        preferred=float(self.task.get("wrist_forward_preferred",0.0))
        forward_err=0.0
        if tm in ("both","left"):
            forward_err += (n["forwardL"]-preferred)**2
        if tm in ("both","right"):
            forward_err += (n["forwardR"]-preferred)**2
        c += float(w.get("wrist_forward",0.0))*forward_err
        c += float(w.get("shoulder_level",0.0))*n["shoulder_height_difference"]**2

        tn=n["tau"]/np.maximum(self.g1.effort,1e-6)
        c += float(w["torque"])*float(tn@tn)
        c += float(w["contact_force"])*float(np.sum(F*F))

        vals=q[self.g1.qpos_indices]
        mid=.5*(self.g1.lower+self.g1.upper)
        half=np.maximum(.5*(self.g1.upper-self.g1.lower),1e-6)
        c += float(w["joint_center"])*float(np.sum(((vals-mid)/half)**2))

        dq=vals-qref[self.g1.qpos_indices]
        c += float(w["previous_pose"])*float(dq@dq)

        pref=float(self.limits["preferred_manipulation_clearance"])
        for a,b in self.clearance_pairs:
            d=self.g1.geom_distance(a,b,0.5)
            if d<pref:
                dd=max(d,1e-4)
                c += float(w["clearance"])*(1/dd-1/pref)**2

        active_specs=[s for _,s in self.active_specs(mode)]
        nullspace=self.g1.contact_nullspace(active_specs)
        manip=self.g1.constrained_manipulability(active_specs,tm,nullspace)
        c -= float(w["manipulability"])*manip
        forward_manip=self.g1.constrained_forward_manipulability(
            active_specs,self.task.get("wrist_forward_heading_body","pelvis"),tm,
            nullspace
        )
        c -= float(w.get("forward_manipulability",0.0))*forward_manip
        return float(c)

    def equality(self,x,mode,target,qref):
        q,F=self.unpack(x,mode)
        n=self.node(q,F,mode)
        out=list(n["residual"][:6])

        for name,_ in self.active_specs(mode):
            out.append(n["gaps"][name])

        out.append(float(np.dot(q[3:7],q[3:7])-1.0))

        if bool(self.task.get("hard_hand_height",True)):
            tm=self.task.get("target_mode","both")
            if tm in ("both","left"): out.append(n["hL"]-target)
            if tm in ("both","right"): out.append(n["hR"]-target)
        return np.asarray(out,float)

    def inequality(self,x,mode,target,qref):
        q,F=self.unpack(x,mode)
        n=self.node(q,F,mode)
        out=[]
        mu=float(self.ground["friction_mu"])
        reserve=float(self.limits["torque_reserve_ratio"])
        min_clear=float(self.limits["minimum_geom_clearance"])
        min_hand_ground=float(self.limits.get("minimum_hand_ground_height",0.02))

        for f in F:
            fx,fy,fz=f
            out += [fz, mu*fz-abs(fx), mu*fz-abs(fy)]

        out.extend((reserve*self.g1.effort-np.abs(n["tau"])).tolist())

        active_names=set(self.modes[mode])
        for name,g in self.all_gaps().items():
            if name not in active_names:
                out.append(g)

        for a,b in self.collision_pairs:
            out.append(self.g1.geom_distance(a,b,0.3)-min_clear)

        wrist_clear=float(self.limits.get("minimum_wrist_body_clearance",0.0))
        for a,b in self.wrist_clearance_pairs:
            out.append(self.g1.geom_distance(a,b,0.5)-wrist_clear)

        shoulder_tol=float(self.limits.get("maximum_shoulder_height_difference",np.inf))
        if np.isfinite(shoulder_tol):
            shoulder_diff=n["shoulder_height_difference"]
            out.extend((shoulder_tol-shoulder_diff,shoulder_tol+shoulder_diff))

        out.append(n["hL"]-min_hand_ground)
        out.append(n["hR"]-min_hand_ground)

        for body, minimum_deg in self.limits.get("minimum_body_pitch_deg", {}).items():
            out.append(self.g1.body_pitch(body)-np.deg2rad(float(minimum_deg)))
        return np.asarray(out,float)

    def solve(self, mode, target, qref):
        x0=self.initial_guess(mode,qref)
        b=self.bounds(mode,qref)
        lb=np.array([a for a,_ in b],float)
        ub=np.array([b for _,b in b],float)

        p=_IpoptStaticProblem(self,mode,target,qref,x0,b)
        nlp=cyipopt.Problem(
            n=len(x0),m=p.m,problem_obj=p,
            lb=lb,ub=ub,cl=p.cl,cu=p.cu
        )
        nlp.add_option("tol",float(self.solver.get("tol",1e-5)))
        nlp.add_option("constr_viol_tol",float(self.solver.get("constr_viol_tol",1e-5)))
        nlp.add_option("acceptable_tol",float(self.solver.get("acceptable_tol",1e-4)))
        nlp.add_option("max_iter",int(self.solver.get("maxiter",250)))
        nlp.add_option("print_level",int(self.solver.get("print_level",0)))
        nlp.add_option("hessian_approximation","limited-memory")
        nlp.add_option("mu_strategy","adaptive")

        x,info=nlp.solve(x0)
        q,F=self.unpack(x,mode)
        n=self.node(q,F,mode)
        self.g1.set_qpos(q)
        active_specs=[s for _,s in self.active_specs(mode)]
        nullspace=self.g1.contact_nullspace(active_specs)
        manip=self.g1.constrained_manipulability(
            active_specs,self.task.get("target_mode","both"),nullspace
        )
        minc=self.min_clearance()
        wrist_clearance=self.min_wrist_body_clearance()
        forward_manip=self.g1.constrained_forward_manipulability(
            active_specs,self.task.get("wrist_forward_heading_body","pelvis"),
            self.task.get("target_mode","both"),nullspace
        )
        body_pitch_deg={
            body:float(np.rad2deg(self.g1.body_pitch(body)))
            for body in self.limits.get("minimum_body_pitch_deg", {})
        }
        ratio=float(np.max(np.abs(n["tau"])/np.maximum(self.g1.effort,1e-6)))
        status=int(info.get("status",-999))

        # Post-hoc physical feasibility check. IPOPT can stop with a non-success
        # status even when the returned point is sufficiently feasible for use
        # as a static pose candidate.
        eq_res = self.equality(x,mode,target,qref)
        iq_res = self.inequality(x,mode,target,qref)
        max_eq = float(np.max(np.abs(eq_res))) if len(eq_res) else 0.0
        min_iq = float(np.min(iq_res)) if len(iq_res) else 0.0

        eq_tol = float(self.solver.get("postcheck_eq_tol", 1e-4))
        iq_tol = float(self.solver.get("postcheck_ineq_tol", 1e-4))
        post_feasible = (max_eq <= eq_tol) and (min_iq >= -iq_tol)

        # Only a point passing every post-check may enter the pose bank.
        ok = post_feasible

        return PoseCandidate(
            target_height=float(target),mode=mode,success=ok,
            status=str(info.get("status_msg","")),
            objective=float(info.get("obj_val",self.objective(x,mode,target,qref))),
            qpos=q.copy(),forces=F.copy(),
            active_contact_names=list(self.modes[mode]),
            left_height=float(n["hL"]),right_height=float(n["hR"]),
            left_forward=float(n["forwardL"]),right_forward=float(n["forwardR"]),
            min_wrist_body_clearance=float(wrist_clearance),
            forward_manipulability=float(forward_manip),
            shoulder_height_difference=float(n["shoulder_height_difference"]),
            torque=n["tau"].copy(),max_torque_ratio=ratio,
            min_clearance=float(minc),manipulability=float(manip),
            body_pitch_deg=body_pitch_deg,
            ipopt_status=status,
            max_eq_violation=max_eq,
            min_ineq_margin=min_iq
        )

class _IpoptStaticProblem:
    def __init__(self,opt,mode,target,qref,x0,bounds):
        self.opt=opt; self.mode=mode; self.target=target; self.qref=qref
        self.rel=float(opt.solver.get("finite_difference_rel_step",2e-5))
        self.abs=float(opt.solver.get("finite_difference_abs_step",1e-7))
        eq=opt.equality(x0,mode,target,qref)
        iq=opt.inequality(x0,mode,target,qref)
        self.neq=len(eq); self.nineq=len(iq); self.m=self.neq+self.nineq
        self.cl=np.r_[np.zeros(self.neq),np.zeros(self.nineq)]
        self.cu=np.r_[np.zeros(self.neq),np.full(self.nineq,np.inf)]
        self.lb=np.array([a for a,_ in bounds],float)
        self.ub=np.array([b for _,b in bounds],float)

    def objective(self,x):
        return self.opt.objective(x,self.mode,self.target,self.qref)

    def constraints(self,x):
        return np.r_[
            self.opt.equality(x,self.mode,self.target,self.qref),
            self.opt.inequality(x,self.mode,self.target,self.qref)
        ]

    def _h(self,v):
        return max(self.abs,self.rel*max(1.0,abs(float(v))))

    def gradient(self,x):
        x=np.asarray(x,float); f0=self.objective(x); g=np.zeros_like(x)
        for j in range(len(x)):
            h=self._h(x[j]); xp=x.copy()
            if x[j]+h<=self.ub[j]:
                xp[j]+=h; g[j]=(self.objective(xp)-f0)/h
            elif x[j]-h>=self.lb[j]:
                xp[j]-=h; g[j]=(f0-self.objective(xp))/h
        return g

    def jacobian(self,x):
        x=np.asarray(x,float); g0=self.constraints(x)
        J=np.zeros((self.m,len(x)))
        for j in range(len(x)):
            h=self._h(x[j]); xp=x.copy()
            if x[j]+h<=self.ub[j]:
                xp[j]+=h; J[:,j]=(self.constraints(xp)-g0)/h
            elif x[j]-h>=self.lb[j]:
                xp[j]-=h; J[:,j]=(g0-self.constraints(xp))/h
        return J.ravel(order="C")
