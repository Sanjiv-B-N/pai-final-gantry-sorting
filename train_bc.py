"""Train a behavior-cloning policy with action chunking (ACT-style) on the demos.

features(obs) -> MLP -> next K actions, each (dx, dy, dz, magnet logit).
Loss: L1 on the motion (as in ACT) + BCE on the magnet command.
At run time the first action of each new chunk is executed and the policy re-infers
every step (n_action_steps = 1, as in Lecture 5); ACT's temporal ensembling is
available as an option.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from policy_env import ACT_DIM, MAX_STEP, PICK_TIMEOUT, GRASP_Z, SAFE_Z

K = 10  # chunk length (1 s at 10 Hz)
FEAT_DIM = 16


def features(obs):
    """Policy input built from the env observation (works on one obs or a batch).

    Only relative geometry, height and the servo tracking error are used (no absolute
    x/y), and every offset is also given clipped to one control step so that small
    errors near a target stand out.
    """
    o = np.atleast_2d(obs)
    z, holding, rel_t, rel_g, dz_place = o[:, 2:3], o[:, 3:4], o[:, 4:6], o[:, 6:8], o[:, 8:9]
    track = o[:, 9:12] / MAX_STEP  # measured - commanded position (is the gripper still moving?)
    dz = np.hstack([dz_place, GRASP_Z - z, SAFE_Z - z])
    clip = lambda v: np.clip(v, -MAX_STEP, MAX_STEP) / MAX_STEP
    f = np.hstack([z, holding, rel_t, clip(rel_t), rel_g, clip(rel_g), clip(dz), track]).astype(np.float32)
    return f if np.ndim(obs) > 1 else f[0]


class ChunkMLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEAT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, K * ACT_DIM))

    def forward(self, x):
        return self.net(x).view(-1, K, ACT_DIM)


def make_chunks(action, episode_index):
    """Target chunk for every frame: the next K actions of the same episode
    (padded by repeating the episode's last action)."""
    out = np.empty((len(action), K, ACT_DIM), dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, episode_index[1:] != episode_index[:-1]])
    ends = np.r_[starts[1:], len(action)]
    for s, e in zip(starts, ends):
        for t in range(s, e):
            idx = np.minimum(np.arange(t, t + K), e - 1)
            out[t] = action[idx]
    return out


class BCPolicy:
    """Deployable policy: features + normalization + model (+ optional temporal ensembling)."""

    def __init__(self, path="policy.pt", ensemble=False, m=0.1):
        ck = torch.load(path, weights_only=False)
        self.model = ChunkMLP()
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.mu, self.sd = ck["mu"], ck["sd"]
        self.ensemble, self.m = ensemble, m
        self.reset()

    def reset(self):
        self.history = []  # (time predicted, chunk)
        self.t = 0
        self.last_chunk = None

    def predict(self, obs):
        x = torch.as_tensor((features(obs) - self.mu) / self.sd, dtype=torch.float32)[None]
        with torch.no_grad():
            y = self.model(x)[0].numpy()
        y[:, :3] *= MAX_STEP
        y[:, 3] = 1 / (1 + np.exp(-y[:, 3]))
        return y

    def __call__(self, obs):
        chunk = self.predict(obs)
        self.last_chunk = chunk
        if not self.ensemble:  # n_action_steps = 1: execute the first action, re-infer next step
            return np.array([*chunk[0, :3], float(chunk[0, 3] > 0.5)], dtype=np.float32)
        self.history = [(t0, c) for t0, c in self.history if self.t - t0 < K] + [(self.t, chunk)]
        # ACT temporal ensembling: w_i = exp(-m * i), i = 0 for the oldest prediction
        acts = np.array([c[self.t - t0] for t0, c in self.history])
        w = np.exp(-self.m * np.arange(len(acts)))
        a = (acts * w[:, None]).sum(0) / w.sum()
        self.t += 1
        return np.array([*a[:3], float(a[3] > 0.5)], dtype=np.float32)


def main(data="data/demos.npz", out="policy.pt", epochs=40, seed=0):
    torch.manual_seed(seed)
    d = np.load(data)
    obs, act, ep = d["obs"], d["action"].copy(), d["episode_index"]
    # data validation: drop demos where the expert itself timed out (e.g. target moved away)
    lengths = np.bincount(ep)
    keep = lengths[ep] < PICK_TIMEOUT
    print(f"dropping {(lengths >= PICK_TIMEOUT).sum()} timed-out demos of {len(np.unique(ep))}")
    obs, act, ep = features(obs[keep]), act[keep], ep[keep]
    act[:, :3] /= MAX_STEP  # motion in [-1, 1]
    chunks = make_chunks(act, ep)

    rng = np.random.default_rng(seed)
    eps = np.unique(ep)
    val_eps = rng.choice(eps, size=len(eps) // 10, replace=False)
    val = np.isin(ep, val_eps)
    mu, sd = obs[~val].mean(0), obs[~val].std(0) + 1e-6
    X = torch.as_tensor((obs - mu) / sd, dtype=torch.float32)
    Y = torch.as_tensor(chunks)
    tr, va = torch.as_tensor(~val), torch.as_tensor(val)
    print(f"train frames {int(tr.sum())}, val frames {int(va.sum())} ({len(val_eps)} held-out episodes)")

    model = ChunkMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bce = nn.BCEWithLogitsLoss()

    def loss_fn(pred, y):
        return (pred[..., :3] - y[..., :3]).abs().mean() + bce(pred[..., 3], y[..., 3])

    Xtr, Ytr = X[tr], Y[tr]
    for e in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            loss = loss_fn(model(Xtr[b]), Ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if (e + 1) % 10 == 0 or e == 0:
            model.eval()
            with torch.no_grad():
                pv = model(X[va])
                err_cm = (pv[:, 0, :3] - Y[va][:, 0, :3]).abs().mean().item() * MAX_STEP * 100
                mag_acc = ((pv[:, 0, 3] > 0) == (Y[va][:, 0, 3] > 0.5)).float().mean().item()
                print(f"epoch {e + 1}: train loss {loss.item():.4f}, val loss {loss_fn(pv, Y[va]).item():.4f}, "
                      f"val first-action err {err_cm:.3f} cm/step, magnet acc {mag_acc * 100:.2f}%", flush=True)
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "K": K}, out)
    print(f"saved {out}")

    # Lecture-5-style debug-policy check: prediction vs ground truth on held-out episodes
    fig, axes = plt.subplots(2, 4, figsize=(14, 5), sharex="col")
    labels = ["dx [m]", "dy [m]", "dz [m]", "magnet"]
    for row, e_id in enumerate(val_eps[:2]):
        m = ep == e_id
        with torch.no_grad():
            p = model(X[torch.as_tensor(m)])[:, 0].numpy()
        p[:, :3] *= MAX_STEP
        p[:, 3] = 1 / (1 + np.exp(-p[:, 3]))
        gt = d["action"][keep][m]
        for j in range(4):
            axes[row, j].plot(gt[:, j], "k-", label="expert (ground truth)")
            axes[row, j].plot(p[:, j], "r--", label="policy prediction")
            axes[row, j].set_title(f"held-out episode {e_id}: {labels[j]}", fontsize=9)
    axes[0, 0].legend(fontsize=8)
    for a in axes[1]:
        a.set_xlabel("step (10 Hz)")
    fig.tight_layout()
    fig.savefig("docs/debug_policy.png", dpi=110)
    print("saved docs/debug_policy.png")


if __name__ == "__main__":
    main(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 40)
