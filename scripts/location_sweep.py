"""Where to put the data: one network, same amount of data, different places.

Trains one PINN on a window [t0, t0 + T] with a fixed small amount of data
(a % of the RK4 reference points in that window) placed in one of 7 layouts:

    even         spread evenly over the window (bin midpoints)
    random       uniform random times (np.random.default_rng(seed).random(n))
    early        first third, evenly (bin midpoints)
    middle       middle third
    late         last third
    first_half   first half
    second_half  second half

Each target time is snapped to the first reference time at or after it, which
reproduces the samples used for the T = 1 and T = 2 runs in
results/location_sweep/location_results.csv exactly.

For t0 > 0 the network is trained in local time s = t - t0 on [0, T], with the
RK4 reference state at t0 (integrated from (1, 1, 1) at dt = 1e-4) as its
initial condition. The Lorenz equations don't depend on t, so this is the same
problem as the window [t0, t0 + T].

Training settings are the tuned ones used for the T = 2 location runs:
4 x 64 tanh network, Adam lr 1e-2 for 10,000 epochs (cosine decay), then
L-BFGS up to 5,000 steps, seed 99, physics on the whole window, output scaling
from the reference path on the window, hard initial condition.

    python location_sweep.py --t0 13 --T 2 --pct 0.25 --layout first_half --out runs/
    python location_sweep.py --plot results.csv --t0 13 --T 2 --out plots/
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

LAYOUTS = ["even", "random", "early", "middle", "late", "first_half", "second_half"]
LABELS = {"even": "spread evenly", "random": "random", "early": "first third",
          "middle": "middle third", "late": "last third", "first_half": "first half",
          "second_half": "second half"}
SEGMENTS = {"even": (0, 1), "early": (0, 1 / 3), "middle": (1 / 3, 2 / 3), "late": (2 / 3, 1),
            "first_half": (0, .5), "second_half": (.5, 1)}
DT = 1e-4


def layout_times(layout, n, T, seed=99):
    """Target times (local, in [0, T]) for n points in a layout."""
    if layout == "random":
        return np.sort(np.random.default_rng(seed).random(n)) * T
    a, b = SEGMENTS[layout]
    a, b = a * T, b * T
    return a + (np.arange(n) + 0.5) * (b - a) / n


def layout_indices(layout, n, T, seed=99):
    """Reference row indices: the first reference time at or after each target."""
    t = np.arange(int(round(T / DT)) + 1) * DT
    return np.unique(np.searchsorted(t, layout_times(layout, n, T, seed) - 1e-12))


def state_at(t0, x0=1., y0=1., z0=1.):
    """RK4 reference state at t0, from (1, 1, 1), dt = 1e-4 (same arithmetic as Lorenz)."""
    from lorenz_pinn_full import Lorenz
    lz = Lorenz(dt=[DT], T=1.0)
    x, y, z = x0, y0, z0
    for _ in range(int(round(t0 / DT))):
        x, y, z = lz.rk4_step(x, y, z, DT)
    return x, y, z


def run(t0, T, pct, layout, seed=99, epochs=10000, lbfgs=5000, out="runs"):
    import tensorflow as tf
    from lorenz_pinn_full import Lorenz_PINN_solver, DTYPE

    class Location_Solver(Lorenz_PINN_solver):
        def __init__(self, layout, n_data, **kw):
            super().__init__(**kw)
            tf.keras.utils.set_random_seed(self.seed)  # pin Keras init too
            self.layout, self.n_data = layout, n_data

        def sample_system(self, dt=DT, n_sample=None):
            ref = self.lorenz_df.loc[self.lorenz_df["dt"] == dt].reset_index(drop=True)
            idx = layout_indices(self.layout, self.n_data, self.T, self.seed)
            self.sample_list.append(ref.iloc[idx].reset_index(drop=True))

        def prepare_training_data(self, dt=DT):
            super().prepare_training_data(dt)
            ref = self.lorenz_df.loc[self.lorenz_df["dt"] == dt, ["x_pos", "y_pos", "z_pos"]]
            self.state_mean = tf.constant(ref.mean().values.reshape(1, 3), dtype=DTYPE)
            self.state_std = tf.constant(ref.std().values.reshape(1, 3), dtype=DTYPE)

    n_ref = int(round(T / DT)) + 1
    n_data = int(round(n_ref * pct / 100))
    x0, y0, z0 = state_at(t0) if t0 > 0 else (1., 1., 1.)
    tag = f"t{t0:g}-{t0 + T:g}" if t0 > 0 else f"T{T:g}"
    name = f"{tag}_pct{pct:g}_{layout}"
    os.makedirs(out, exist_ok=True)
    start = time.time()
    s = Location_Solver(layout, n_data, dt=[DT], method="rk4", optimizer="adam",
                        learning_rate=1e-2, lbfgs_iters=lbfgs, T=T, seed=seed,
                        x0=x0, y0=y0, z0=z0)
    s.pinn_pipeline(dt=DT, epochs=epochs, print_every=5000)
    p = s.pred_df
    ref3 = np.sqrt((p[["x_pos", "y_pos", "z_pos"]] ** 2).sum(axis=1))
    t_s = s.sample_list[0]["t"].values
    row = dict(name=name, t0=t0, T=T, layout=layout, pct_reference=pct, n_data=len(t_s),
               n_reference=n_ref, t_first=round(t0 + t_s[0], 4), t_last=round(t0 + t_s[-1], 4),
               max_gap=float(np.diff(np.r_[0, t_s, T]).max()),
               rel_l2=float(np.linalg.norm(p["err_3d"]) / np.linalg.norm(ref3)),
               max_abs=float(p[["x_err", "y_err", "z_err"]].abs().values.max()),
               final_loss=s.loss_history[-1]["total"],
               lbfgs_steps=len(s.loss_history) - s.first_stage_epochs,
               seed=seed, seconds=round(time.time() - start))
    out_p = p[["t", "x_pos", "y_pos", "z_pos", "x_pinn", "y_pinn", "z_pinn"]].copy()
    out_p["t"] = out_p["t"] + t0
    out_p.iloc[::10].to_csv(os.path.join(out, f"pred_{name}.csv"), index=False, float_format="%.10g")
    with open(os.path.join(out, f"row_{name}.json"), "w") as f:
        json.dump(row, f)
    print(json.dumps(row))
    return row


def plot(csv, t0, T, out, flips=(), ref_x_csv=None, pct_show=0.25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv)
    df = df[(df["t0"] == t0) & (df["T"] == T)] if "t0" in df else df[df["T"] == T]
    n_ref = int(round(T / DT)) + 1
    pcts = sorted(df["pct_reference"].unique())
    colors = {pcts[0]: "tab:blue", pcts[-1]: "tab:orange"}
    win = f"[{t0:g}, {t0 + T:g}]"

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6.6))
    ys = np.arange(len(LAYOUTS))[::-1]
    for k, pct in enumerate(pcts):
        sub = df[df["pct_reference"] == pct].set_index("layout")
        n = int(sub["n_data"].iloc[0])
        y = [ys[LAYOUTS.index(l)] + (0.15 if k == 0 else -0.15) for l in sub.index]
        ax.scatter(sub["rel_l2"], y, s=80, color=colors[pct], zorder=3,
                   label=f"{pct:g}% of reference ({n} pts)")
    ax.set_xscale("log")
    ax.set_xlim(5e-7, 2)
    ax.axvline(1e-4, color="gray", ls="--", lw=1)
    ax.set_yticks(ys, [LABELS[l] for l in LAYOUTS])
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="x", ls=":", color="gray")
    for y in ys:
        ax.axhline(y, color="0.9", lw=1, zorder=0)
    ax.set_xlabel(f"relative L2 error over all of {win} (dashed: 1e-4 target)", fontsize=11)
    ax.set_title(f"Error by where the data sits (one network, t in {win})")
    ax.legend(loc="lower center")

    # right panel: where the pct_show points sit, over the reference x(t)
    n_show = int(round(n_ref * pct_show / 100))
    for l in LAYOUTS:
        t = t0 + layout_indices(l, n_show, T) * DT
        ax2.vlines(t, ys[LAYOUTS.index(l)] - 0.12, ys[LAYOUTS.index(l)] + 0.12,
                   color="tab:orange", lw=1.2)
    ax2.set_yticks(ys, [LABELS[l] for l in LAYOUTS])
    ax2.tick_params(axis="y", labelsize=12)
    ax2.set_ylim(-0.5, len(LAYOUTS) - 0.5)
    ax2.set_xlim(t0, t0 + T)
    ax2.set_xlabel("t")
    axr = ax2.twinx()
    if ref_x_csv:
        r = pd.read_csv(ref_x_csv)
        r = r[(r["t"] >= t0 - 1e-9) & (r["t"] <= t0 + T + 1e-9)]
        xc = "x" if "x" in r else "x_pos"
        axr.plot(r["t"], r[xc], color="0.7", lw=1.2, zorder=0)
        axr.axhline(0, color="0.85", lw=0.8, zorder=0)
    axr.set_ylabel("reference x(t)", color="gray")
    for i, tf_ in enumerate(flips):
        ax2.axvline(tf_, color="tab:red", ls="--", lw=1.2)
        ax2.text(tf_, len(LAYOUTS) - 0.48, f" flip {i + 1}\n t ≈ {tf_:.3f}", color="tab:red",
                 va="top", ha="left", fontsize=10)
    ax2.set_title(f"Where the {pct_show:g}% data points sit (gray: reference x, red: x flips sign)")
    fig.suptitle(f"One network on {win}: same amount of data, different places. "
                 f"Data = {pcts[0]:g}% or {pcts[-1]:g}% of the {n_ref:,} RK4 reference points",
                 fontsize=13)
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"location_t{t0:g}-{t0 + T:g}.png")
    fig.savefig(path, dpi=120)
    print("saved", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--pct", type=float, default=0.25)
    p.add_argument("--layout", choices=LAYOUTS)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--lbfgs", type=int, default=5000)
    p.add_argument("--out", default="runs")
    p.add_argument("--plot", metavar="CSV")
    p.add_argument("--flips", type=float, nargs="*", default=[])
    p.add_argument("--ref-x", help="CSV with columns t and x (or x_pos) for the gray reference curve")
    a = p.parse_args()
    if a.plot:
        plot(a.plot, a.t0, a.T, a.out, a.flips, a.ref_x)
    else:
        run(a.t0, a.T, a.pct, a.layout, a.seed, lbfgs=a.lbfgs, out=a.out)
