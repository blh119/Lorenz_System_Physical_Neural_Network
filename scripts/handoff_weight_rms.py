"""
One number per handoff for the hidden weights: warm start vs fresh weights.

Reads two time-marching runs saved with --save-dir (one warm, one with
--no-warm-start) and draws one plain graph for each:

    weight_rms_warm.png    warm start (each window starts from the previous window's weights)
    weight_rms_fresh.png   a new random network in every window

The number is the RMS of every weight and bias in the 4 hidden layers of 64
units (the output layer is left out), divided by the RMS of the random network
the run started from. So 1.0 means "the same size as a newly made network" and
3.0 means "three times bigger". It is taken from each window's final weights,
which are the weights at that window's handoff.

A window counts as off the rails when its end point is more than --fail-err
away from the reference (3D distance). Such windows are drawn in red, and the
last point before the first one is the "last good handoff".

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
        rows.append(dict(window=int(os.path.basename(p)[7:9]) + 1, start=start, handoff_t=end,
                         rms_start=hidden_rms(d, "start"), rms_handoff=hidden_rms(d, "final"),
                         handoff_err_3d=err))
    return pd.DataFrame(rows)


def draw(df, base, title, color, path, ymax, fail_err):
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
    t = np.r_[0.0, df["handoff_t"].values]
    y = np.r_[df["rms_start"].iloc[0], df["rms_handoff"].values] / base
    bad = np.r_[False, df["handoff_err_3d"].values > fail_err]
    for tv in t[1:]:
        ax.axvline(tv, color="0.5", lw=0.5, ls=":", zorder=0)
    ax.axhline(1.0, color="0.5", lw=1, ls="--")
    ax.text(t[-1] + 0.08, 1.0, "new random\nnetwork", va="center", fontsize=8, color="0.35")
    ax.plot(t, y, color=color, lw=2, zorder=2)
    ax.scatter(t[~bad], y[~bad], color=color, s=36, zorder=3)
    label = [ti == int(ti) for ti in t]        # label whole-number handoffs only
    if bad.any():
        first = int(np.argmax(bad))
        ax.scatter(t[bad], y[bad], color="#d62728", s=60, marker="X", zorder=4,
                   label="window went off the rails")
        ax.scatter([t[first - 1]], [y[first - 1]], s=140, facecolors="none", edgecolors="black",
                   lw=1.5, zorder=4, label=f"last good handoff (t = {t[first - 1]:g})")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        label[first - 1] = True
    for ti, yi, show in zip(t, y, label):
        if show:
            ax.annotate(f"{yi:.2f}", (ti, yi), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=9)
    ax.set_xticks(np.arange(0, t[-1] + 0.01, 1.0))
    ax.set_xlim(-0.2, t[-1] + 0.45)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("t (dotted lines are handoffs)")
    ax.set_ylabel("hidden-weight size\n(RMS, 1 = new random network)")
    ax.set_title(title, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
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
    draw(warm, base, "Warm start: each window starts from the previous window's weights",
         "#c23b8a", os.path.join(a.out, "weight_rms_warm.png"), ymax, a.fail_err)
    draw(fresh, base, "No warm start: a new random network in every window",
         "#2e9e6b", os.path.join(a.out, "weight_rms_fresh.png"), ymax, a.fail_err)
    out = pd.concat([warm.assign(run="warm"), fresh.assign(run="fresh")])
    out.to_csv(os.path.join(a.out, "weight_rms.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
