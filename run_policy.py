"""Closed-loop evaluation and demo video for the learned policy.

  python run_policy.py eval [n_layouts]   # policy vs scripted expert on the same unseen layouts
  python run_policy.py video              # demo.mp4 with the learned policy in control
"""
import sys
import numpy as np
import cv2

from sort_sim import Runner, IMG, data
from policy_env import GantryEnv, run_episode, expert_controller, NAMES, SUBSTEPS
from train_bc import BCPolicy
from sort_sim import build_perception


def policy_controller(policy):
    def make(env):
        policy.reset()
        return lambda: policy(env.obs())
    return make


def evaluate(n=50, seed=123):
    rng = np.random.default_rng(seed)
    perceive = build_perception(rng)
    results = {}
    for name, ctrl in [("expert", expert_controller),
                       ("policy", policy_controller(BCPolicy())),
                       ("policy (temporal ens.)", policy_controller(BCPolicy(ensemble=True)))]:
        for disturbed in (False, True):
            env = GantryEnv(perceive, np.random.default_rng(seed + 1))  # identical layouts per controller
            sorted_cubes = timeouts = perfect = 0
            for _ in range(n):
                env.reset()
                picks, ok, s = run_episode(env, ctrl, rng=env.rng, disturb_at=1 if disturbed else None)
                sorted_cubes += s
                timeouts += picks - ok
                perfect += s == len(NAMES)
            key = f"{name}{' + disturbance' if disturbed else ''}"
            results[key] = (sorted_cubes, timeouts, perfect)
            print(f"{key:38s} cubes sorted {sorted_cubes}/{n * len(NAMES)} "
                  f"({100 * sorted_cubes / (n * len(NAMES)):.1f}%), perfect layouts {perfect}/{n}, "
                  f"timed-out picks {timeouts}", flush=True)
    return results


def world_to_inset(x, y, z):
    s = 2 * (1.2 - z) * np.tan(np.deg2rad(20)) / IMG
    return int(2 * (x / s + IMG / 2 - 0.5)), int(2 * (-y / s + IMG / 2 - 0.5))


def video(seed=7, episodes=2):
    rng = np.random.default_rng(seed)
    perceive = build_perception(rng)
    policy = BCPolicy()
    env = GantryEnv(perceive, rng)
    run = Runner("demo.mp4")
    every = max(1, run.every)
    state = {"k": 0, "base": None}

    def on_substep():
        state["k"] += 1
        if state["k"] % every:
            return
        if state["base"] is not None:
            view = state["base"].copy()
            sp = env.sp()
            if policy.last_chunk is not None and env.target is not None:
                pts = sp + np.cumsum(policy.last_chunk[:, :3], axis=0)
                for p in pts:
                    cv2.circle(view, world_to_inset(*p), 4, (60, 60, 255), -1)
            cv2.circle(view, world_to_inset(*env.ee()), 7, (255, 255, 255), -1)
            cv2.circle(view, world_to_inset(*env.ee()), 7, (0, 0, 0), 2)
            run.inset = view
        run.writer.append_data(run.frame())

    def on_pick(img, cls, dets):
        run.set_ai_view(img, cls, dets)
        state["base"] = run.inset.copy()
        policy.last_chunk = None
        run.caption = f"{len(dets)} cubes left -> learned policy picks nearest & stacks it" if dets else \
            "All cubes sorted"

    env.on_substep = on_substep
    for ep in range(episodes):
        env.reset()
        run.inset, state["base"] = None, None
        disturbed = ep == 1
        run.header = (f"Episode {ep + 1}/{episodes} | LEARNED POLICY in control (BC + action chunking) | 1x"
                      + (" | disturbance test" if disturbed else ""))
        run.caption = "Perceiving scene with learned segmentation model..."
        env.wait(0.8)
        _, _, s = run_episode(env, policy_controller(policy), rng=rng,
                              disturb_at=1 if disturbed else None, on_pick=on_pick,
                              on_disturb=lambda n: setattr(run, "caption",
                                                           f"Disturbance! {n} was moved -> policy re-perceives next pick"))
        run.caption = f"Episode {ep + 1} done: {s}/{len(NAMES)} cubes sorted and stacked by the learned policy"
        env.wait(1.5)
        print(f"video episode {ep + 1}: {s}/{len(NAMES)} sorted", flush=True)
    end_card(run)
    run.close()


# numbers from `python run_policy.py eval 50` (docs/eval_results.txt)
END_LINES = [
    ("Closed-loop evaluation: 50 unseen random layouts (300 cubes)", "", 1.0),
    ("", "", 0.8),
    ("Scripted expert (teacher)", "300/300 sorted", 0.9),
    ("Learned policy (BC, first action)", "300/300 sorted", 0.9),
    ("  + random disturbance", "300/300 sorted", 0.9),
    ("Policy + ACT temporal ensembling", "183/300 sorted (61%)", 0.9),
    ("", "", 0.8),
    ("~4% of policy pick attempts timed out and were retried after re-perception", "", 0.75),
]


def end_card(run, seconds=6):
    card = np.zeros((720, 1280, 3), dtype=np.uint8)
    y = 200
    for label, value, scale in END_LINES:
        cv2.putText(card, label, (110, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        cv2.putText(card, value, (760, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        y += 55
    for _ in range(int(seconds * 30)):
        run.writer.append_data(card[..., ::-1])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "eval":
        evaluate(int(sys.argv[2]) if len(sys.argv) > 2 else 50)
    else:
        video()
