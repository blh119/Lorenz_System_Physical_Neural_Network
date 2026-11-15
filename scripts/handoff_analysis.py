"""
What happens to the network at each window handoff: warm start vs fresh weights.

Reads two time-marching runs made with --save-dir (one warm, one with
--no-warm-start) and their --save-predictions tables, and draws six graphs:

    handoff_errors.png    error along the path, and the error left at each handoff
    handoff_losses.png    loss at the start of each window, after Adam, after L-BFGS
    starting_guess.png    what each network predicts before training, at three handoffs
    weight_size.png       how big each layer's weights are at the end of each window
    unit_health.png       saturated tanh units and working units per layer
    weight_movement.png   how far training moves the weights, and when

It also writes handoff_summary.csv with one row per window and run.

The network is re-evaluated here in NumPy from the saved weights (4 tanh layers,
then a linear output layer), so nothing is retrained.

Example (the run used for the graphs):
    python time_marching.py --T 5 --window 0.25 --save-dir runs/warm \\
        --save-predictions runs/pred_warm.csv
    python time_marching.py --T 5 --window 0.25 --no-warm-start --save-dir runs/fresh \\
        --save-predictions runs/pred_fresh.csv
    python handoff_analysis.py --warm runs/warm --fresh runs/fresh \\
        --warm-pred runs/pred_warm.csv --fresh-pred runs/pred_fresh.csv --out results/handoff_weights
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUNS = {"warm": dict(color="#c23b8a", label="warm start (weights copied from the previous window)"),
        "fresh": dict(color="#2e9e6b", label="fresh weights every window")}
SNAPS = ["start", "epoch1", "epoch10", "epoch100", "epoch1000", "epoch2000", "final"]
SAT = 2.0          # |tanh input| above this counts as saturated (slope below 0.07)
N_TAU = 1000       # evenly spaced inputs on [-1, 1] used to probe the hidden units


# ----------------------------------------------------------------------
# Loading and re-running the network
# ----------------------------------------------------------------------

def load_run(folder):
    """List of dicts, one per window, with its weights (by snapshot) and settings."""
    windows = []
    for path in sorted(glob.glob(os.path.join(folder, "window_*.npz"))):
        k = int(os.path.basename(path)[7:9])
        d = np.load(path)
        w = {"k": k, "start": float(d["start"]), "end": float(d["end"]),
             "ic": d["ic"], "std": d["state_std"], "W": {}}
        for s in SNAPS:
            w["W"][s] = [d[f"w_{s}_{i}"] for i in range(10) if f"w_{s}_{i}" in d.files] or None
        w["losses"] = pd.read_csv(os.path.join(folder, f"losses_{k:02d}.csv"))
        windows.append(w)
    return windows


def forward(weights, tau):
    """Network output and the input to each tanh layer, for inputs tau (n,)."""
    h = tau.reshape(-1, 1)
    pre = []
    for i in range(0, len(weights) - 2, 2):
        z = h @ weights[i] + weights[i + 1]
        pre.append(z)
        h = np.tanh(z)
    return h @ weights[-2] + weights[-1], pre


def predict(weights, w, t_local):
    """Physical (x, y, z) at local times, with the hard initial condition, as in predict_state."""
    L = w["end"] - w["start"]
    out, _ = forward(weights, 2.0 * t_local / L - 1.0)
    return w["ic"] + (t_local / L)[:, None] * out * w["std"]


def flat(weights):
    return np.concatenate([a.ravel() for a in weights])


# ----------------------------------------------------------------------
# One row per window
# ----------------------------------------------------------------------

def summarize(name, windows, pred, epochs=10000):
    tau = np.linspace(-1, 1, N_TAU)
    rows = []
    for w in windows:
        lh = w["losses"]
        adam_end = lh[lh["epoch"] <= epochs]  # L-BFGS steps are numbered after the Adam epochs
        ends = pred[np.isclose(pred["t"], w["end"], atol=1e-7)]
        row = {"run": name, "window": w["k"], "start": w["start"], "end": w["end"],
               "loss_epoch1": lh["total"].iloc[0], "data_loss_epoch1": lh["data"].iloc[0],
               "loss_end_adam": adam_end["total"].iloc[-1], "loss_final": lh["total"].iloc[-1],
               "physics_final": lh["physics"].iloc[-1], "data_final": lh["data"].iloc[-1],
               "handoff_err_3d": float(ends["err_3d"].iloc[-1]) if len(ends) else np.nan}
        f0, ff = flat(w["W"]["start"]), flat(w["W"]["final"])
        row["norm_start"] = np.linalg.norm(f0)
        row["norm_final"] = np.linalg.norm(ff)
        row["moved"] = np.linalg.norm(ff - f0)
        row["moved_rel"] = row["moved"] / row["norm_start"]
        for s in SNAPS[1:-1]:
            if w["W"][s] is not None:
                row[f"moved_{s}"] = np.linalg.norm(flat(w["W"][s]) - f0)
        final = w["W"]["final"]
        for i in range(0, len(final), 2):
            row[f"kernel_norm_L{i // 2 + 1}"] = np.linalg.norm(final[i])
            row[f"kernel_max_L{i // 2 + 1}"] = np.abs(final[i]).max()
        _, pre = forward(final, tau)
        for j, z in enumerate(pre, start=1):
            row[f"saturated_L{j}"] = float((np.abs(z) > SAT).mean())
            sv = np.linalg.svd(np.tanh(z), compute_uv=False)
            row[f"working_units_L{j}"] = int((sv > 0.01 * sv[0]).sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Graphs
# ----------------------------------------------------------------------

def _handoffs(ax, windows, every=1):
    for w in windows[1::every]:
        ax.axvline(w["start"], color="gray", lw=0.4, ls=":", zorder=0)


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def plot_errors(preds, summ, windows, out):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for name, p in preds.items():
        d = p.iloc[::10]
        a1.plot(d["t"], d["err_3d"] + 1e-12, color=RUNS[name]["color"], lw=1, label=RUNS[name]["label"])
        s = summ[summ.run == name]
        a2.plot(s["end"], s["handoff_err_3d"], marker="o", ms=5, lw=1.3, color=RUNS[name]["color"],
                label=RUNS[name]["label"])
    for ax in (a1, a2):
        ax.set_yscale("log")
        ax.axhline(1e-4, color="gray", ls="--", lw=1)
        _handoffs(ax, windows)
        ax.grid(alpha=0.25, which="major")
    a1.set_ylabel("3D error along the path")
    a1.set_title("Error along the path (dotted lines are handoffs; dashed line is the 1e-4 target)")
    a1.legend(loc="upper left", fontsize=9)
    a2.set_ylabel("3D error at the end of the window")
    a2.set_title("Error handed to the next window at each handoff")
    a2.set_xlabel("t")
    _save(fig, out, "handoff_errors.png")


def plot_losses(summ, windows, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    cols = [("loss_epoch1", "Loss at epoch 1 (the starting guess)"),
            ("loss_end_adam", "Loss after 10,000 Adam epochs"),
            ("loss_final", "Loss after L-BFGS (final)")]
    for ax, (c, title) in zip(axes, cols):
        for name in RUNS:
            s = summ[summ.run == name]
            ax.plot(s["start"], s[c], marker="o", ms=4, lw=1.2, color=RUNS[name]["color"],
                    label=RUNS[name]["label"])
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("window start time")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("total loss")
    axes[0].legend(fontsize=8, loc="best")
    _save(fig, out, "handoff_losses.png")


def plot_starting_guess(runs, preds, out, picks):
    """Before any training: the copied network's curve vs the fresh one's vs the truth."""
    fig, axes = plt.subplots(3, len(picks), figsize=(4.4 * len(picks), 8), sharex="col")
    ref = preds["fresh"]
    for c, k in enumerate(picks):
        ww, wf = runs["warm"][k], runs["fresh"][k]
        L = ww["end"] - ww["start"]
        t = np.linspace(0, L, 300)
        truth = ref[(ref["t"] >= ww["start"] - 1e-9) & (ref["t"] <= ww["end"] + 1e-9)]
        guess = {"warm": predict(ww["W"]["start"], ww, t), "fresh": predict(wf["W"]["start"], wf, t)}
        for r, v in enumerate("xyz"):
            ax = axes[r, c]
            ax.plot(truth["t"], truth[f"{v}_pos"], color="black", lw=2, label="reference (RK4)")
            for name in RUNS:
                ax.plot(ww["start"] + t, guess[name][:, r], color=RUNS[name]["color"], lw=1.4, ls="--",
                        label=f"{name}: before training")
            if r == 0:
                ax.set_title(f"window {k + 1}: t = {ww['start']:g} to {ww['end']:g}", fontsize=11)
            if c == 0:
                ax.set_ylabel(v)
            ax.grid(alpha=0.25)
        axes[-1, c].set_xlabel("t")
    axes[0, 0].legend(fontsize=8, loc="best")
    _save(fig, out, "starting_guess.png")


def plot_weight_size(summ, out):
    layers = sorted(int(c.split("L")[1]) for c in summ.columns if c.startswith("kernel_norm_L"))
    fig, axes = plt.subplots(1, len(layers) + 1, figsize=(3.3 * (len(layers) + 1), 3.8), sharex=True)
    for ax, j in zip(axes, layers):
        for name in RUNS:
            s = summ[summ.run == name]
            ax.plot(s["start"], s[f"kernel_norm_L{j}"], marker="o", ms=3, lw=1.2, color=RUNS[name]["color"])
        ax.set_title("output layer" if j == layers[-1] else f"hidden layer {j}", fontsize=10)
        ax.set_xlabel("window start")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("size of the layer's weights (norm)")
    ax = axes[-1]
    for name in RUNS:
        s = summ[summ.run == name]
        ax.plot(s["start"], s["norm_final"], marker="o", ms=3, lw=1.2, color=RUNS[name]["color"],
                label=name)
    ax.set_title("all 12,803 weights", fontsize=10)
    ax.set_xlabel("window start")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.suptitle("Weight size at the end of each window", y=1.02)
    _save(fig, out, "weight_size.png")


def plot_unit_health(summ, out):
    layers = sorted(int(c.split("L")[1]) for c in summ.columns if c.startswith("saturated_L"))
    fig, axes = plt.subplots(2, len(layers), figsize=(3.6 * len(layers), 6.4), sharex=True)
    for j, (a1, a2) in zip(layers, axes.T):
        for name in RUNS:
            s = summ[summ.run == name]
            a1.plot(s["start"], 100 * s[f"saturated_L{j}"], marker="o", ms=3, lw=1.2,
                    color=RUNS[name]["color"], label=name)
            a2.plot(s["start"], s[f"working_units_L{j}"], marker="o", ms=3, lw=1.2,
                    color=RUNS[name]["color"], label=name)
        a1.set_title(f"hidden layer {j}", fontsize=10)
        a2.set_xlabel("window start")
        a2.set_ylim(0, max(12, summ[f"working_units_L{j}"].max() + 1))
        for a in (a1, a2):
            a.grid(alpha=0.25)
    axes[0, 0].set_ylabel("% saturated (|input| > 2)")
    axes[1, 0].set_ylabel("distinct unit shapes (rank)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Health of the hidden units at the end of each window", y=1.01)
    _save(fig, out, "unit_health.png")


def plot_movement(summ, out, n_weights):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 4.6))
    for name in RUNS:
        s = summ[summ.run == name]
        # Adam's first step moves every weight by about the learning rate (1e-2) unless its
        # gradient is close to zero, so a full first step has length 1e-2 * sqrt(n_weights).
        full = 1e-2 * np.sqrt(n_weights)
        a3.plot(s["start"], s["moved_epoch1"] / full, marker="o", ms=4, lw=1.2,
                color=RUNS[name]["color"], label=name)
        a1.plot(s["start"], s["moved"], marker="o", ms=4, lw=1.2, color=RUNS[name]["color"],
                label=RUNS[name]["label"])
        steps = [c for c in ["moved_epoch1", "moved_epoch10", "moved_epoch100", "moved_epoch1000",
                             "moved_epoch2000", "moved"] if c in s]
        frac = s[steps].div(s["moved"], axis=0).median()
        a2.plot(range(len(steps)), frac.values, marker="o", ms=6, lw=1.6, color=RUNS[name]["color"],
                label=name)
    a1.set_title("Distance the weights move during each window", fontsize=11)
    a1.set_xlabel("window start")
    a1.set_ylabel("distance from start to final weights")
    a1.legend(fontsize=8)
    a2.set_xticks(range(6), ["1 step", "10", "100", "1,000\n(physics on)", "2,000\n(physics full)", "end"])
    a2.set_ylabel("distance so far / total distance")
    a2.set_title("When the moving happens (median over windows)", fontsize=11)
    a2.legend(fontsize=8)
    a3.set_title("Size of Adam's first step (1 = every weight moves)", fontsize=11)
    a3.set_xlabel("window start")
    a3.set_ylabel("first step / full first step")
    a3.set_ylim(0, 1.1)
    a3.legend(fontsize=8)
    for a in (a1, a2, a3):
        a.grid(alpha=0.25)
    _save(fig, out, "weight_movement.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--warm", required=True, help="--save-dir of the warm-start run")
    p.add_argument("--fresh", required=True, help="--save-dir of the --no-warm-start run")
    p.add_argument("--warm-pred", required=True)
    p.add_argument("--fresh-pred", required=True)
    p.add_argument("--out", default="handoff_weights")
    p.add_argument("--epochs", type=int, default=10000, help="Adam epochs per window in the runs")
    p.add_argument("--picks", default="1,7,13", help="0-based windows for starting_guess.png")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = {"warm": load_run(a.warm), "fresh": load_run(a.fresh)}
    preds = {"warm": pd.read_csv(a.warm_pred), "fresh": pd.read_csv(a.fresh_pred)}
    summ = pd.concat([summarize(n, runs[n], preds[n], a.epochs) for n in RUNS], ignore_index=True)
    summ.to_csv(os.path.join(a.out, "handoff_summary.csv"), index=False)
    longest = max(runs.values(), key=len)
    n = min(len(runs["warm"]), len(runs["fresh"]))
    picks = [k for k in map(int, a.picks.split(",")) if k < n]

    plot_errors(preds, summ, longest, a.out)
    plot_losses(summ, longest, a.out)
    plot_starting_guess(runs, preds, a.out, picks)
    plot_weight_size(summ, a.out)
    plot_unit_health(summ, a.out)
    plot_movement(summ, a.out, len(flat(runs["warm"][0]["W"]["final"])))
