import numpy as np

def _clip(a, lo=-1.0, hi=1.0):
    return np.clip(a, lo, hi).astype(np.float32)


class ScriptedPickPlacePolicy:
    """
    RL-Franka-Panda style (no fingertip sites, no grasp detection),
    but stabilized:
    - NEVER chase obj_z for vertical targets (use obj_z0 from episode start)
    - gentler descend/press to avoid contact explosion
    """

    def __init__(
        self,
        kp_xy=6.0,
        kp_z=6.0,
        xy_tol=0.012,

        hover_height=0.10,      # above object (EE center)
        grasp_depth=0.006,      # go slightly BELOW obj center (small!)
        lift_height=0.14,       # lift above table
        place_height=0.03,      # above goal before open

        close_steps=35,
        open_steps=20,

        max_steps_per_phase=220,
        descend_steps=140,

        # contact stabilization
        press_dz=-0.08,         # gentler press during CLOSE
        z_safety_max=0.60,      # cap EE target z to avoid "chasing flight"

        verbose=False,
    ):
        self.kp_xy = kp_xy
        self.kp_z = kp_z
        self.xy_tol = xy_tol

        self.hover_height = hover_height
        self.grasp_depth = grasp_depth
        self.lift_height = lift_height
        self.place_height = place_height

        self.close_steps = close_steps
        self.open_steps = open_steps

        self.max_steps_per_phase = max_steps_per_phase
        self.descend_steps = descend_steps

        self.press_dz = press_dz
        self.z_safety_max = z_safety_max

        self.verbose = verbose
        self.reset()

    def reset(self):
        self.phase = "HOVER_OBJ"
        self.phase_step = 0

        self.obj_z0 = None          # object "table" height baseline (episode start)
        self.lock_obj_xy = None     # freeze xy while descending/closing
        self.lock_goal_xy = None

    def step_phase_counter(self):
        self.phase_step += 1

    def _ee_pos(self, obs):
        return obs["observation"][:3].astype(np.float32)

    def _goto(self, ee, target, grip):
        # cap z target to avoid chasing explosions
        target = target.copy()
        target[2] = float(np.clip(target[2], -1e9, self.z_safety_max))

        d = target - ee
        a = np.zeros(4, dtype=np.float32)
        a[0] = self.kp_xy * d[0]
        a[1] = self.kp_xy * d[1]
        a[2] = self.kp_z  * d[2]
        a[3] = grip
        return _clip(a)

    def _bump(self, nxt, obj=None, goal=None):
        if self.verbose:
            print(f"[Policy] {self.phase} -> {nxt}")
        self.phase = nxt
        self.phase_step = 0

        # freeze XY at phase transitions (prevents "single finger dragging" chase)
        if obj is not None and nxt in ("DESCEND", "CLOSE", "LIFT"):
            self.lock_obj_xy = np.array([obj[0], obj[1]], dtype=np.float32)
        if goal is not None and nxt in ("MOVE_TO_GOAL", "PLACE_DESCEND"):
            self.lock_goal_xy = np.array([goal[0], goal[1]], dtype=np.float32)

    def act(self, obs):
        ee = self._ee_pos(obs)
        obj = obs["achieved_goal"].astype(np.float32)
        goal = obs["desired_goal"].astype(np.float32)

        if self.obj_z0 is None:
            # baseline: object at reset (on table)
            self.obj_z0 = float(obj[2])

        # XY distances
        dist_xy_obj = np.linalg.norm((ee[:2] - obj[:2]))
        dist_xy_goal = np.linalg.norm((ee[:2] - goal[:2]))

        # Use baseline Z for all vertical targets (IMPORTANT!)
        hover_z = self.obj_z0 + self.hover_height
        grasp_z = self.obj_z0 - self.grasp_depth
        lift_z  = self.obj_z0 + self.lift_height

        # ------------------------------------------------
        if self.phase == "HOVER_OBJ":
            target = np.array([obj[0], obj[1], hover_z], dtype=np.float32)
            a = self._goto(ee, target, grip=+1.0)
            if (dist_xy_obj < self.xy_tol and abs(ee[2] - hover_z) < 0.02) or self.phase_step > self.max_steps_per_phase:
                self._bump("DESCEND", obj=obj)
            return a

        if self.phase == "DESCEND":
            xy = self.lock_obj_xy if self.lock_obj_xy is not None else obj[:2]
            target = np.array([xy[0], xy[1], grasp_z], dtype=np.float32)
            a = self._goto(ee, target, grip=+1.0)

            if self.verbose and self.phase_step % 10 == 0:
                print(f"[DESCEND] step={self.phase_step:03d} ee_z={ee[2]:.4f} tgt_z={grasp_z:.4f} obj_z={obj[2]:.4f}")

            if abs(ee[2] - grasp_z) < 0.01 or self.phase_step > self.descend_steps:
                self._bump("CLOSE", obj=obj)
            return a

        if self.phase == "CLOSE":
            # close + gentle downward press (do NOT over-press)
            a = np.array([0.0, 0.0, self.press_dz, -1.0], dtype=np.float32)
            a = _clip(a)
            if self.phase_step >= self.close_steps:
                self._bump("LIFT", obj=obj)
            return a

        if self.phase == "LIFT":
            xy = self.lock_obj_xy if self.lock_obj_xy is not None else obj[:2]
            target = np.array([xy[0], xy[1], lift_z], dtype=np.float32)
            a = self._goto(ee, target, grip=-1.0)

            if self.verbose and self.phase_step % 15 == 0:
                print(f"[LIFT] obj_z_delta={obj[2] - self.obj_z0:+.4f} ee_z={ee[2]:.4f} tgt_z={lift_z:.4f}")

            if abs(ee[2] - lift_z) < 0.03 or self.phase_step > self.max_steps_per_phase:
                self._bump("MOVE_TO_GOAL", goal=goal)
            return a

        if self.phase == "MOVE_TO_GOAL":
            xy = self.lock_goal_xy if self.lock_goal_xy is not None else goal[:2]
            target = np.array([xy[0], xy[1], lift_z], dtype=np.float32)
            a = self._goto(ee, target, grip=-1.0)

            if (dist_xy_goal < self.xy_tol) or self.phase_step > self.max_steps_per_phase:
                self._bump("PLACE_DESCEND", goal=goal)
            return a

        if self.phase == "PLACE_DESCEND":
            xy = self.lock_goal_xy if self.lock_goal_xy is not None else goal[:2]
            target = np.array([xy[0], xy[1], self.obj_z0 + self.place_height], dtype=np.float32)
            a = self._goto(ee, target, grip=-1.0)
            if abs(ee[2] - target[2]) < 0.02 or self.phase_step > self.descend_steps:
                self._bump("OPEN")
            return a

        if self.phase == "OPEN":
            a = np.array([0.0, 0.0, 0.0, +1.0], dtype=np.float32)
            if self.phase_step >= self.open_steps:
                self._bump("RETREAT")
            return a

        if self.phase == "RETREAT":
            a = np.array([0.0, 0.0, +0.5, +1.0], dtype=np.float32)
            a = _clip(a)
            if self.phase_step >= 25:
                return np.array([0.0, 0.0, 0.0, +1.0], dtype=np.float32)
            return a

        return np.array([0.0, 0.0, 0.0, +1.0], dtype=np.float32)
