"""
One number per handoff for the hidden weights: warm start vs fresh weights.

Reads two time-marching runs saved with --save-dir (one warm, one with
--no-warm-start) and draws one plain graph for each:

    weight_rms_warm.png    warm start (each window starts from the previous window's weights)
    weight_rms_fresh.png   a new random network in every window
    weight_rms_both.png    both on one graph

The number is the RMS of every weight and bias in the 4 hidden layers of 64
units (the output layer is left out), divided by the RMS of the random network
the run started from. So 1.0 means "the same size as a newly made network" and
3.0 means "three times bigger". It is taken from each window's final weights,
which are the weights at that window's handoff.

A window counts as off the rails when its end point is more than --fail-err
away from the reference (3D distance). Such windows are drawn in red, and the
last point before the first one is the "last good handoff".

Under each graph is the training loss of the window that ends at that handoff:
its final loss (after L-BFGS, the network that is handed over), from the runs'
losses_XX.csv files.

Example (the runs used for the graphs):
    python time_marching.py --T 5 --window 0.25 --save-dir runs/warm --save-predictions runs/pred_warm.csv
    python time_marching.py --T 5 --window 0.25 --no-warm-start --save-dir runs/fresh \\
        --save-predictions runs/pred_fresh.csv
    python handoff_weight_rms.py --warm runs/warm --fresh runs/fresh --warm-pred runs/pred_warm.csv \\
        --fresh-pred runs/pred_fresh.csv --out results/handoff_weights/simple
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_HIDDEN_ARRAYS = 8   # kernels and biases of the 4 hidden layers (w_0 .. w_7)
ADAM_EPOCHS = 10000   # Adam epochs per window; the rows after these are L-BFGS steps


def hidden_rms(d, snap):
    w = np.concatenate([d[f"w_{snap}_{i}"].ravel() for i in range(N_HIDDEN_ARRAYS)])
    return float(np.sqrt(np.mean(w ** 2)))


def load(folder, pred_csv):
    rows = []
    paths = sorted(glob.glob(os.path.join(folder, "window_*.npz")))
    ref = pd.read_csv(pred_csv) if pred_csv else None
    for p in paths:
        d = np.load(p)
        start, end = float(d["start"]), float(d["end"])
        err = np.nan
        if ref is not None:
            k = int(os.path.basename(p)[7:9])
            err = float(ref.loc[ref["window"] == k, "err_3d"].iloc[-1])
        k = int(os.path.basename(p)[7:9])
        losses = pd.read_csv(os.path.join(folder, f"losses_{k:02d}.csv"))
        adam = losses.loc[losses["epoch"] <= ADAM_EPOCHS, "total"]
        rows.append(dict(window=k + 1, start=start, handoff_t=end,
                         rms_start=hidden_rms(d, "start"), rms_handoff=hidden_rms(d, "final"),
                         handoff_err_3d=err, loss_after_adam=float(adam.iloc[-1]),
                         loss_handoff=float(losses["total"].iloc[-1])))
    return pd.DataFrame(rows)


def draw(runs, base, title, path, ymax, fail_err):
    """Top: hidden-weight size at each handoff. Bottom: loss at each handoff, same t axis.

    runs: list of (df, label, color). The red X's and the last-good circle are drawn
    for any run that has windows that went off the rails.
    """
    fig, (ax, axl) = plt.subplots(2, 1, figsize=(11, 7.2), dpi=150, sharex=True,
                                  gridspec_kw=dict(height_ratios=[2, 1.3]))
    t_all = np.r_[0.0, runs[0][0]["handoff_t"].values]
    for a in (ax, axl):
        for tv in t_all[1:]:
            a.axvline(tv, color="0.5", lw=0.5, ls=":", zorder=0)
        a.spines[["top", "right"]].set_visible(False)
    ax.axhline(1.0, color="0.5", lw=1, ls="--")
    ax.text(t_all[-1] + 0.08, 1.0, "new random\nnetwork", va="center", fontsize=8, color="0.35")
    marked = False
    for df, label, color in runs:
        t = np.r_[0.0, df["handoff_t"].values]
        y = np.r_[df["rms_start"].iloc[0], df["rms_handoff"].values] / base
        bad = np.r_[False, df["handoff_err_3d"].values > fail_err]
        ax.plot(t, y, color=color, lw=2, zorder=2, label=label)
        ax.scatter(t[~bad], y[~bad], color=color, s=32, zorder=3)
        show = [ti == int(ti) for ti in t]
        if len(runs) > 1:
            show[0] = False
        th, lh = df["handoff_t"].values, df["loss_handoff"].values
        bad_h = bad[1:]
        axl.plot(th, lh, color=color, lw=2, zorder=2)
        axl.scatter(th[~bad_h], lh[~bad_h], color=color, s=28, zorder=3)
        if bad.any():
            first = int(np.argmax(bad))
            ax.scatter(t[bad], y[bad], color="#d62728", s=60, marker="X", zorder=4,
                       label=None if marked else "window went off the rails")
            ax.scatter([t[first - 1]], [y[first - 1]], s=140, facecolors="none", edgecolors="black",
                       lw=1.5, zorder=4, label=f"last good handoff (t = {t[first - 1]:g})")
            axl.scatter(th[bad_h], lh[bad_h], color="#d62728", s=60, marker="X", zorder=4)
            axl.scatter([th[first - 2]], [lh[first - 2]], s=140, facecolors="none", edgecolors="black",
                        lw=1.5, zorder=4)
            show[first - 1] = True
            marked = True
        for ti, yi, sh in zip(t, y, show):
            if sh:
                ax.annotate(f"{yi:.2f}", (ti, yi), textcoords="offset points", xytext=(0, 11),
                            ha="center", va="center", fontsize=9, color=color)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("hidden-weight size\n(RMS, 1 = new random network)")
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    axl.set_yscale("log")
    axl.set_ylim(1e-11, 1e2)
    axl.set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0])
    axl.set_ylabel("final training loss\nat the handoff")
    axl.set_xticks(np.arange(0, t_all[-1] + 0.01, 1.0))
    axl.set_xlim(-0.2, t_all[-1] + 0.45)
    axl.set_xlabel("t (dotted lines are handoffs)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", required=True)
    p.add_argument("--fresh", required=True)
    p.add_argument("--warm-pred")
    p.add_argument("--fresh-pred")
    p.add_argument("--out", required=True)
    p.add_argument("--fail-err", type=float, default=1e-2)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    warm, fresh = load(a.warm, a.warm_pred), load(a.fresh, a.fresh_pred)
    base = warm["rms_start"].iloc[0]           # both runs share window 1's random start
    for df in (warm, fresh):
        df["rms_handoff_norm"] = df["rms_handoff"] / base
        df["rms_start_norm"] = df["rms_start"] / base
    ymax = 1.2 * max(warm["rms_handoff_norm"].max(), fresh["rms_handoff_norm"].max(), 1.0)
    warm_run = (warm, "warm start (weights copied from the previous window)", "#c23b8a")
    fresh_run = (fresh, "no warm start (new random network every window)", "#2e9e6b")
    draw([warm_run], base, "Warm start: each window starts from the previous window's weights",
         os.path.join(a.out, "weight_rms_warm.png"), ymax, a.fail_err)
    draw([fresh_run], base, "No warm start: a new random network in every window",
         os.path.join(a.out, "weight_rms_fresh.png"), ymax, a.fail_err)
    draw([warm_run, fresh_run], base,
         "Hidden-weight size at each handoff: warm start vs a new network every window",
         os.path.join(a.out, "weight_rms_both.png"), ymax, a.fail_err)
    out = pd.concat([warm.assign(run="warm"), fresh.assign(run="fresh")])
    out.to_csv(os.path.join(a.out, "weight_rms.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
