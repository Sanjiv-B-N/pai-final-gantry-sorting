"""Policy-facing environment for the gantry sorting task.

Reuses the scene and the learned perception model from sort_sim.py and exposes a
10 Hz action interface:  action = (dx, dy, dz, magnet)  where dx..dz move the
gripper set-point (clipped to MAX_STEP per step) and magnet in {0, 1}.

One "pick" = take the perceived target cube to its color's pad and release it.
Task sequencing (which cube next) and parking the gantry for the camera stay
rule-based; the pick-and-place motion itself is what the policy learns.
"""
import numpy as np
import mujoco

from sort_sim import (model, data, NAMES, COLORS, GOALS, CUBE, PARK, DONE_X, SPAWN_X, SPAWN_Y, cube_ids, ee_id,
                      color_of, set_cube, reset, random_points, top_image, build_perception)

# Gravity compensation on the vertical axis (a real gantry uses a counterweight or
# brake); without it the z position controller sags ~6 mm below its set-point.
Z_DOF = model.jnt_dofadr[model.joint("jz").id]
GRAV_COMP = model.body_subtreemass[model.body("tool").id] * -model.opt.gravity[2]

# Stiffer position servos than the scripted demo so the gripper tracks its set-point
# closely at 10 Hz control (kp, kv per axis).
model.actuator_gainprm[:, 0] = 2000
model.actuator_biasprm[:, 1] = -2000
model.actuator_biasprm[:, 2] = -200

CTRL_DT = 0.1
SUBSTEPS = int(round(CTRL_DT / model.opt.timestep))
MAX_STEP = 0.045  # m per control step (0.45 m/s)
SAFE_Z = 0.30
GRASP_Z = 2 * CUBE + 0.002
PICK_TIMEOUT = 150  # control steps
OBS_DIM = 12
ACT_DIM = 4


_CAM = None


def shared_camera():
    """One top-camera renderer for the whole process: on macOS, creating/freeing a
    second offscreen renderer corrupted the images of the next one."""
    global _CAM
    if _CAM is None:
        _CAM = mujoco.Renderer(model, 160, 160)
    return _CAM


def place_z(level):
    return 2 * CUBE * (level + 1) + 0.008


class GantryEnv:
    def __init__(self, perceive, rng):
        self.perceive = perceive
        self.rng = rng
        self.cam = shared_camera()
        self.on_substep = None  # optional hook, e.g. video recording

    # ----- episode / pick bookkeeping -----
    def reset(self, layout=None):
        reset(random_points(self.rng, len(NAMES)) if layout is None else layout)
        self.held = None
        self.magnet = 0.0
        self.stack = {c: 0 for c in COLORS}
        self.target = None

    def ee(self):
        return data.site_xpos[ee_id].copy()

    def sp(self):
        """Commanded gripper position (set-point), i.e. the robot's proprioception."""
        return data.ctrl + np.array([0, 0, 0.10])

    def detect(self):
        """Perceive with the learned model (gantry must be parked out of view)."""
        img = top_image(self.cam)
        dets, cls = self.perceive(img)
        return dets, img, cls

    def begin_pick(self, color, xy):
        self.target = (color, np.asarray(xy, dtype=float))
        self.released = False

    def obs(self):
        """[set-point xyz, holding, target-sp xy, goal-sp xy, place height-sp z,
        tracking error (measured - commanded) xyz]"""
        e = self.sp()
        c, t = self.target
        g = np.array(GOALS[c])
        return np.array([e[0], e[1], e[2], float(self.held is not None),
                         t[0] - e[0], t[1] - e[1], g[0] - e[0], g[1] - e[1],
                         place_z(self.stack[c]) - e[2], *(self.ee() - e)], dtype=np.float32)

    def pick_done(self):
        return self.released and self.ee()[2] > SAFE_Z - 0.05

    # ----- physics -----
    def _substep(self):
        if self.held:
            set_cube(self.held, *(data.site_xpos[ee_id] - [0, 0, CUBE]))
        data.qfrc_applied[Z_DOF] = GRAV_COMP
        mujoco.mj_step(model, data)
        if self.on_substep:
            self.on_substep()

    def _update_magnet(self, magnet):
        e = self.ee()
        if magnet > 0.5 and self.held is None:
            for n in NAMES:
                p = data.xpos[cube_ids[n]]
                if np.linalg.norm(p[:2] - e[:2]) < 0.03 and 0 <= e[2] - (p[2] + CUBE) < 0.015:
                    self.held = n
                    break
        elif magnet <= 0.5 and self.held is not None:
            c = self.target[0]
            self.stack[c] += int(color_of(self.held) == c)
            self.held = None
            self.released = True
        self.magnet = float(magnet > 0.5)

    def step(self, action):
        a = np.asarray(action, dtype=float)
        start = data.ctrl.copy()
        goal = start + np.clip(a[:3], -MAX_STEP, MAX_STEP)
        goal[2] = np.clip(goal[2], -0.58, 0.3)
        for i in range(SUBSTEPS):
            data.ctrl[:] = start + (goal - start) * (i + 1) / SUBSTEPS
            self._substep()
        self._update_magnet(a[3])

    def move_ctrl(self, ctrl, speed=0.45):
        """Scripted joint-space move (used only for parking between picks)."""
        start = data.ctrl.copy()
        goal = np.asarray(ctrl, dtype=float)
        n = max(1, int(np.linalg.norm(goal - start) / speed / model.opt.timestep))
        for i in range(n):
            s = (i + 1) / n
            data.ctrl[:] = start + (goal - start) * (3 * s**2 - 2 * s**3)
            self._substep()

    def wait(self, t):
        for _ in range(int(t / model.opt.timestep)):
            self._substep()

    # ----- evaluation -----
    def sorted_count(self):
        return sum(np.linalg.norm(data.xpos[cube_ids[n]][:2] - GOALS[color_of(n)]) < 0.06 for n in NAMES)

    def waiting_cubes(self):
        return [n for n in NAMES if n != self.held and data.xpos[cube_ids[n]][0] < DONE_X]


class Expert:
    """Scripted demonstrator: waypoint follower emitting 10 Hz actions."""

    def __init__(self, env):
        self.env = env

    def start(self):
        c, t = self.env.target
        g = GOALS[c]
        self.plan = [((t[0], t[1], SAFE_Z), 0), ((t[0], t[1], GRASP_Z), 0), (None, 1),
                     ((t[0], t[1], SAFE_Z), 1), ((g[0], g[1], SAFE_Z), 1),
                     ((g[0], g[1], place_z(self.env.stack[c])), 1), (None, 0),
                     ((g[0], g[1], SAFE_Z), 0)]
        self.i = 0
        self.tries = 0

    def act(self):
        sp, ee = self.env.sp(), self.env.ee()
        while self.i < len(self.plan):
            wp, mag = self.plan[self.i]
            if wp is None:  # magnet switch step: let the gripper settle, switch, give up after 3 tries
                done = (self.env.held is not None) if mag == 1 else self.env.released
                if done or self.tries >= 3:
                    self.i += 1
                    self.tries = 0
                    continue
                if np.linalg.norm(ee - sp) > 0.004 and self.tries == 0:
                    return np.array([0, 0, 0, 1 - mag], dtype=np.float32)  # still settling
                self.tries += 1
                return np.array([0, 0, 0, mag], dtype=np.float32)
            d = np.array(wp) - sp
            if np.linalg.norm(d) < 1e-3:
                self.i += 1
                continue
            return np.array([*np.clip(d, -MAX_STEP, MAX_STEP), mag], dtype=np.float32)
        return np.array([0, 0, 0, 0], dtype=np.float32)


def run_episode(env, controller, rng=None, disturb_at=None, on_pick=None, on_disturb=None):
    """Sort all cubes of one random layout. controller(env) -> callable returning actions.

    Returns (picks attempted, picks that ended in time, cubes sorted at the end).
    """
    picks = ok_picks = 0
    for k in range(len(NAMES) + 3):
        env.move_ctrl(PARK)
        dets, img, cls = env.detect()
        if on_pick:
            on_pick(img, cls, dets)
        if not dets:
            break
        e = env.ee()[:2]
        c, xy, _ = min(dets, key=lambda d: np.linalg.norm(d[1] - e))
        env.begin_pick(c, xy)
        policy = controller(env)
        picks += 1
        for t in range(PICK_TIMEOUT):
            env.step(policy())
            if disturb_at == k and t == 30 and rng is not None:
                moved = disturb(env, rng)
                if moved and on_disturb:
                    on_disturb(moved)
            if env.pick_done():
                ok_picks += 1
                break
        else:
            if env.held:  # drop whatever is still held
                env._update_magnet(0)
    return picks, ok_picks, env.sorted_count()


def disturb(env, rng):
    tx = env.target[1]  # move a cube other than the one currently being picked
    waiting = [n for n in env.waiting_cubes()
               if n != env.held and np.linalg.norm(data.xpos[cube_ids[n]][:2] - tx) > 0.05]
    if not waiting:
        return None
    n = waiting[rng.integers(len(waiting))]
    others = np.array([data.xpos[cube_ids[m]][:2] for m in NAMES if m != n])
    for _ in range(1000):  # free spot >= 10 cm from every other cube
        p = np.array([rng.uniform(*SPAWN_X), rng.uniform(*SPAWN_Y)])
        if np.linalg.norm(others - p, axis=1).min() > 0.1:
            break
    else:
        return None
    set_cube(n, *p)
    mujoco.mj_forward(model, data)
    return n


def expert_controller(env):
    ex = Expert(env)
    ex.start()
    return ex.act
