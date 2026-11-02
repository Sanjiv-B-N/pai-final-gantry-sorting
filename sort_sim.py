"""Vision-guided color sorting & stacking with a gantry robot in MuJoCo.

A learned perception model (pixel-wise MLP segmentation trained on synthetic
top-camera images) detects every colored cube. The gantry robot re-perceives the
scene before every pick (closed loop), picks the nearest cube with a magnetic
gripper and stacks it on the goal pad of the same color. Mid-task, a cube is
moved by an external "disturbance" to show the robot adapts.
"""
import sys
import numpy as np
import mujoco
import imageio
import cv2
from scipy import ndimage
from sklearn.neural_network import MLPClassifier

COLORS = {"red": "0.9 0.15 0.15 1", "green": "0.15 0.8 0.2 1", "blue": "0.15 0.3 0.95 1"}
BGR = {"red": (40, 40, 230), "green": (50, 200, 50), "blue": (240, 80, 40)}
GOALS = {"red": (0.30, -0.30), "green": (0.30, 0.0), "blue": (0.30, 0.30)}
N_PER = 2
CUBE = 0.025  # half size
SPAWN_X = (-0.32, 0.08)
SPAWN_Y = (-0.32, 0.32)
DONE_X = 0.2  # detections beyond this x are on goal pads
IMG = 160
PARK = (0.45, 0.42, 0.0)
NAMES = [f"{c}{i}" for c in COLORS for i in range(N_PER)]
color_of = lambda name: name.rstrip("0123456789")

cubes_xml = "".join(
    f'<body name="{n}" pos="0 0 {CUBE}"><freejoint/>'
    f'<geom type="box" size="{CUBE} {CUBE} {CUBE}" rgba="{COLORS[color_of(n)]}" mass="0.05"/></body>'
    for n in NAMES)
goals_xml = "".join(
    f'<geom type="cylinder" size="0.06 0.001" pos="{x} {y} 0.001" '
    f'rgba="{COLORS[c][:-1]} 0.35" contype="0" conaffinity="0"/>'
    for c, (x, y) in GOALS.items())

XML = f"""
<mujoco>
  <option timestep="0.002"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".85 .85 .8" rgb2=".75 .75 .7" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="6 6"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse=".8 .8 .8"/>
    <light pos="1 -1 2" dir="-1 1 -2" diffuse=".4 .4 .4"/>
    <geom name="floor" type="plane" size="0.6 0.6 0.01" material="grid"/>
    {goals_xml}
    <camera name="top" pos="0 0 1.2" xyaxes="1 0 0 0 1 0" fovy="40"/>
    <camera name="view" pos="-0.25 -1.15 0.95" xyaxes="1 -0.2 0 0.12 0.6 0.8" fovy="45"/>
    <geom type="box" size="0.02 0.45 0.3" pos="-0.5 0 0.3" rgba=".3 .3 .35 1"/>
    <geom type="box" size="0.02 0.45 0.3" pos="0.5 0 0.3" rgba=".3 .3 .35 1"/>
    <body name="bridge" pos="0 0 0.62">
      <joint name="jx" type="slide" axis="1 0 0" range="-0.45 0.45"/>
      <geom type="box" size="0.03 0.45 0.02" rgba=".5 .5 .55 1" contype="0" conaffinity="0" mass="1"/>
      <body name="carriage">
        <joint name="jy" type="slide" axis="0 1 0" range="-0.42 0.42"/>
        <geom type="box" size="0.05 0.05 0.03" rgba=".9 .6 .1 1" contype="0" conaffinity="0" mass="0.5"/>
        <body name="tool">
          <joint name="jz" type="slide" axis="0 0 1" range="-0.58 0.3"/>
          <geom type="cylinder" size="0.012 0.25" pos="0 0 -0.25" rgba=".7 .7 .7 1" contype="0" conaffinity="0" mass="0.2"/>
          <geom type="cylinder" size="0.03 0.01" pos="0 0 -0.51" rgba=".2 .2 .2 1" contype="0" conaffinity="0" mass="0.05"/>
          <site name="ee" pos="0 0 -0.52"/>
        </body>
      </body>
    </body>
    {cubes_xml}
  </worldbody>
  <actuator>
    <position joint="jx" kp="400" kv="60"/>
    <position joint="jy" kp="400" kv="60"/>
    <position joint="jz" kp="400" kv="60"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
cube_ids = {n: model.body(n).id for n in NAMES}
cube_qadr = {n: model.jnt_qposadr[model.body(n).jntadr[0]] for n in NAMES}
cube_vadr = {n: model.jnt_dofadr[model.body(n).jntadr[0]] for n in NAMES}
ee_id = model.site("ee").id


def random_points(rng, k, avoid=()):
    while True:
        pts = np.column_stack([rng.uniform(*SPAWN_X, k), rng.uniform(*SPAWN_Y, k)])
        allp = np.vstack([pts, *avoid]) if len(avoid) else pts
        d = np.linalg.norm(allp[:, None] - allp[None], axis=-1) + np.eye(len(allp))
        if d.min() > 0.1:
            return pts


def set_cube(n, x, y, z=CUBE):
    a, v = cube_qadr[n], cube_vadr[n]
    data.qpos[a:a + 7] = [x, y, z, 1, 0, 0, 0]
    data.qvel[v:v + 6] = 0


def reset(layout):
    mujoco.mj_resetData(model, data)
    for n, (x, y) in zip(NAMES, layout):
        set_cube(n, x, y)
    data.qpos[:3] = data.ctrl[:] = PARK  # gantry out of the top camera's view
    mujoco.mj_forward(model, data)


# ---------- Perception: learned pixel segmentation -> instance detections ----------
def top_image(r):
    r.update_scene(data, camera="top")
    return r.render()


def pixel_to_world(v, u):
    # top camera looks straight down; scale at the cube top-face height
    h = 1.2 - 2 * CUBE
    s = 2 * h * np.tan(np.deg2rad(40) / 2) / IMG
    return np.array([(u - IMG / 2 + 0.5) * s, -(v - IMG / 2 + 0.5) * s])


def build_perception(rng, n_train=60, n_test=40):
    """Train a pixel-wise segmentation net (RGB -> background/red/green/blue).

    Labels come for free from MuJoCo's segmentation renderer; at run time only the
    RGB image is used: segment -> connected components -> world xy per instance.
    """
    r = mujoco.Renderer(model, IMG, IMG)
    seg = mujoco.Renderer(model, IMG, IMG)
    seg.enable_segmentation_rendering()
    cls_ids = list(COLORS)
    lut = np.zeros(model.ngeom + 1, dtype=int)
    for n in NAMES:
        lut[model.body(n).geomadr[0]] = cls_ids.index(color_of(n)) + 1

    def sample(n):
        imgs, labels, layouts = [], [], []
        for _ in range(n):
            layout = random_points(rng, len(NAMES))
            reset(layout)
            imgs.append(top_image(r))
            seg.update_scene(data, camera="top")
            labels.append(lut[seg.render()[..., 0]])
            layouts.append(layout)
        return np.array(imgs), np.array(labels), np.array(layouts)

    imgs, labels, _ = sample(n_train)
    X = imgs.reshape(-1, 3).astype(np.float32) / 255.0
    yl = labels.ravel()
    keep = (yl > 0) | (rng.random(yl.size) < 0.1)  # subsample background
    net = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=200, random_state=0)
    net.fit(X[keep], yl[keep])

    def perceive(img):
        cls = net.predict(img.reshape(-1, 3).astype(np.float32) / 255.0).reshape(IMG, IMG)
        dets = []
        for k, c in enumerate(cls_ids, start=1):
            comp, n = ndimage.label(cls == k)
            for i in range(1, n + 1):
                vs, us = np.nonzero(comp == i)
                if len(vs) < 20:
                    continue
                xy = pixel_to_world(np.median(vs), np.median(us))
                if xy[0] < DONE_X:
                    dets.append((c, xy, (np.median(us), np.median(vs))))
        return dets, cls

    t_imgs, t_labels, t_layouts = sample(n_test)
    acc = np.mean([(net.predict(i.reshape(-1, 3) / 255.0) == l.ravel()).mean() for i, l in zip(t_imgs, t_labels)])
    errs, found = [], 0
    for img, layout in zip(t_imgs, t_layouts):
        dets, _ = perceive(img)
        found += len(dets)
        for n, xy in zip(NAMES, layout):
            same = [d[1] for d in dets if d[0] == color_of(n)]
            errs.append(min(np.linalg.norm(s - xy) for s in same) if same else np.inf)
    errs = np.array(errs)
    print(f"perception: pixel acc {acc*100:.2f}%, detections {found}/{len(errs)}, "
          f"position err mean {errs.mean()*100:.2f} cm, max {errs.max()*100:.2f} cm", flush=True)
    r.close()
    seg.close()
    return perceive


# ---------- Control + video ----------
def ee_target_to_ctrl(x, y, z):
    # gantry joint frame: tool tip is 0.62 - 0.52 = 0.10 m above floor at jz = 0
    return np.array([x, y, z - 0.10])


class Runner:
    def __init__(self, video_path, fps=30):
        self.r = mujoco.Renderer(model, 720, 1280)
        self.writer = imageio.get_writer(video_path, fps=fps, quality=8)
        self.every = int(round(1 / (fps * model.opt.timestep)))
        self.held = None
        self.k = 0
        self.caption = ""
        self.header = ""
        self.inset = None

    def set_ai_view(self, img, cls, dets):
        view = img[..., ::-1].copy()
        for k, c in enumerate(COLORS, start=1):
            view[cls == k] = (0.4 * view[cls == k] + 0.6 * np.array(BGR[c])).astype(np.uint8)
        view = cv2.resize(view, (320, 320), interpolation=cv2.INTER_NEAREST)
        for _, _, (u, v) in dets:
            cv2.circle(view, (int(u * 2), int(v * 2)), 16, (0, 220, 255), 3)
        cv2.rectangle(view, (0, 0), (319, 319), (255, 255, 255), 2)
        self.inset = view

    def frame(self):
        self.r.update_scene(data, camera="view")
        f = self.r.render()[..., ::-1].copy()
        if self.inset is not None:
            f[50:370, 20:340] = self.inset
            cv2.putText(f, "AI view: learned segmentation", (20, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(f, self.header, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.rectangle(f, (0, 670), (1280, 720), (0, 0, 0), -1)
        cv2.putText(f, self.caption, (20, 703), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        return f[..., ::-1]

    def step(self):
        if self.held:
            set_cube(self.held, *(data.site_xpos[ee_id] - [0, 0, CUBE]))
        mujoco.mj_step(model, data)
        if self.k % self.every == 0:
            self.writer.append_data(self.frame())
        self.k += 1

    def move(self, x, y, z, speed=0.45, ctrl=None):
        start = data.ctrl.copy()
        goal = ee_target_to_ctrl(x, y, z) if ctrl is None else np.array(ctrl)
        steps = max(1, int(np.linalg.norm(goal - start) / speed / model.opt.timestep))
        for i in range(steps):
            s = (i + 1) / steps
            data.ctrl[:] = start + (goal - start) * (3 * s**2 - 2 * s**3)
            self.step()
        self.wait(0.2)

    def wait(self, t):
        for _ in range(int(t / model.opt.timestep)):
            self.step()

    def grasp(self):
        ee = data.site_xpos[ee_id]
        dists = {n: np.linalg.norm(data.xpos[cube_ids[n]][:2] - ee[:2]) for n in NAMES}
        n = min(dists, key=dists.get)
        if dists[n] < 0.04:
            self.held = n
        return self.held

    def close(self):
        self.writer.close()
        self.r.close()


def disturb(run, rng):
    """Externally move one cube that is still waiting to be sorted."""
    waiting = [n for n in NAMES if n != run.held and data.xpos[cube_ids[n]][0] < DONE_X]
    if not waiting:
        return None
    n = waiting[rng.integers(len(waiting))]
    others = [data.xpos[cube_ids[m]][:2][None] for m in NAMES if m != n]
    set_cube(n, *random_points(rng, 1, others)[0])
    mujoco.mj_forward(model, data)
    return n


def main(episodes=2, seed=1):
    rng = np.random.default_rng(seed)
    perceive = build_perception(rng)
    cam = mujoco.Renderer(model, IMG, IMG)
    run = Runner("demo.mp4")
    ok_total = 0
    for ep in range(episodes):
        reset(random_points(rng, len(NAMES)))
        run.held, run.inset = None, None
        stack = {c: 0 for c in COLORS}
        run.header = f"Episode {ep + 1}/{episodes}  |  random layout  |  1x speed"
        run.caption = "Perceiving scene with learned segmentation model..."
        run.wait(0.8)
        for pick in range(len(NAMES) + 2):
            img = top_image(cam)
            dets, cls = perceive(img)
            run.set_ai_view(img, cls, dets)
            if not dets:
                break
            ee = data.site_xpos[ee_id][:2]
            c, (px, py), _ = min(dets, key=lambda d: np.linalg.norm(d[1] - ee))
            run.caption = f"{len(dets)} cubes left -> pick nearest: {c} at ({px:+.2f}, {py:+.2f}) m"
            run.wait(0.4)
            run.move(px, py, 0.3)
            run.move(px, py, 2 * CUBE + 0.002)
            held = run.grasp()
            run.move(px, py, 0.3)
            if pick == 1 and (moved := disturb(run, rng)):
                run.caption = f"Disturbance! {moved} was moved -> robot will re-perceive"
            else:
                run.caption = f"Stacking {c} cube on {c} pad (level {stack[c] + 1})"
            gx, gy = GOALS[c]
            run.move(gx, gy, 0.3)
            run.move(gx, gy, 2 * CUBE * (stack[c] + 1) + 0.008)
            run.held = None
            stack[c] += 1
            run.wait(0.2)
            run.move(gx, gy, 0.3)
            run.move(0, 0, 0, ctrl=PARK)  # clear the camera view before next perception
            print(f"ep{ep} pick{pick}: target={c} est=({px:.3f},{py:.3f}) grasped={held}", flush=True)
        ok = sum(np.linalg.norm(data.xpos[cube_ids[n]][:2] - GOALS[color_of(n)]) < 0.06 for n in NAMES)
        ok_total += ok
        run.caption = f"Episode {ep + 1} done: {ok}/{len(NAMES)} cubes sorted and stacked"
        run.wait(1.5)
    run.close()
    print(f"sorted correctly: {ok_total}/{len(NAMES)*episodes}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
