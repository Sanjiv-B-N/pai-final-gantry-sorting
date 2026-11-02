"""Collect expert demonstrations for behavior cloning.

The scripted expert plays the role of the teleoperator. Following DART, the robot
executes the expert action plus Gaussian noise (scaled by commanded speed) while the *clean* expert action is
recorded as the label, so the data also covers how to recover from small errors.
One episode in the dataset = one pick-and-place.
"""
import sys
import numpy as np
from sort_sim import build_perception
from policy_env import GantryEnv, Expert, run_episode, MAX_STEP

NOISE_MAX = 0.015  # m, per-axis std upper bound (per-pick std ~ U(0, NOISE_MAX))


def main(n_layouts=300, out="data/demos.npz", seed=0):
    rng = np.random.default_rng(seed)
    env = GantryEnv(build_perception(rng), rng)
    obs, act, ep_idx, frame_idx = [], [], [], []
    ep = 0

    def controller(env):
        nonlocal ep
        ex = Expert(env)
        ex.start()
        sigma = rng.uniform(0, NOISE_MAX)
        t = 0

        def step():
            nonlocal t
            o, a = env.obs(), ex.act()
            obs.append(o)
            act.append(a)
            ep_idx.append(ep)
            frame_idx.append(t)
            t += 1
            noisy = a.copy()
            # noise scales with commanded speed: none while holding still / switching magnet
            scale = min(1.0, np.linalg.norm(a[:3]) / MAX_STEP)
            noisy[:3] = np.clip(a[:3] + rng.normal(0, sigma, 3) * scale, -MAX_STEP, MAX_STEP)
            return noisy
        ep += 1
        return step

    sorted_total = 0
    for i in range(n_layouts):
        env.reset()
        _, _, s = run_episode(env, controller, rng=rng, disturb_at=1 if i % 3 == 0 else None)
        sorted_total += s
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{n_layouts} layouts, {ep} episodes, {len(obs)} frames, "
                  f"expert sorted {sorted_total}/{6 * (i + 1)}", flush=True)
    np.savez_compressed(out, obs=np.array(obs), action=np.array(act),
                        episode_index=np.array(ep_idx), frame_index=np.array(frame_idx))
    print(f"saved {out}: {ep} episodes, {len(obs)} frames")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
