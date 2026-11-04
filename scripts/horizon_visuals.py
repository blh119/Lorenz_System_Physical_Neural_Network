"""
Plots for the prediction-horizon study (one network vs time-marching).

Reads the prediction tables that lorenz_pinn_full / interval_validation /
time_marching write (columns t, x_pos, y_pos, z_pos, x_pinn, y_pinn, z_pinn,
x_err, y_err, z_err, err_3d), so nothing has to be retrained to make them.

    from horizon_visuals import Horizon_Visuals
    hv = Horizon_Visuals(save_dir="results/horizon/plots")
    hv.plot_states_compare("pred_T3_s99.csv", "pred_tm_T3_s99.csv", T=3)
    hv.plot_errors_compare("pred_T3_s99.csv", "pred_tm_T3_s99.csv", T=3)
    hv.plot_unit_importance(T=5)
    plot_unit_placement("unit_placement.csv", save_dir="results/horizon/plots")
    plot_window_lengths("results/horizon", T=10, save_dir="results/horizon/plots")

Or from the command line, to make every plot for which predictions exist:

    python horizon_visuals.py --pred-dir results/horizon --save-dir results/horizon/plots
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Horizon_Visuals:
    """Side-by-side plots of one network vs time-marching, plus where in each unit data matters."""

    VARS = ["x", "y", "z"]
    COLORS = {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}
    METHODS = ["one network", "time-marching (1-unit windows)"]

    def __init__(self, save_dir=None, every=10):
        """
        save_dir : folder for the PNGs (None: don't save)
        every    : plot every k-th time point (the tables have 10,000 rows per unit)
        """
        self.save_dir = save_dir
        self.every = every
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _load(self, pred):
        df = pd.read_csv(pred) if isinstance(pred, str) else pred
        return df.iloc[::self.every].reset_index(drop=True)

    def _save(self, fig, name):
        if self.save_dir:
            fig.savefig(os.path.join(self.save_dir, name), dpi=150, bbox_inches="tight")

    @staticmethod
    def _windows(ax, T, window=1.0):
        for k in np.arange(window, T, window):
            ax.axvline(k, color="gray", lw=0.7, ls=":")

    @staticmethod
    def _rel_l2(df):
        ref = np.sqrt((df[["x_pos", "y_pos", "z_pos"]] ** 2).sum(axis=1))
        return np.linalg.norm(df["err_3d"]) / np.linalg.norm(ref)

    # ------------------------------------------------------------------
    # 1. x, y, z: reference vs PINN, one network next to time-marching
    # ------------------------------------------------------------------

    def plot_states_compare(self, one_pred, tm_pred, T, save_name=None):
        """3 rows (x, y, z) x 2 columns (one network | time-marching).

        Either prediction may be None (e.g. one network was not run at T = 5).
        """
        preds = [None if p is None else self._load(p) for p in (one_pred, tm_pred)]
        fig, axes = plt.subplots(3, 2, figsize=(12, 7.5), sharex=True, sharey="row")
        for col, (df, method) in enumerate(zip(preds, self.METHODS)):
            for row, v in enumerate(self.VARS):
                ax = axes[row, col]
                if df is None:
                    ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", color="gray")
                    continue
                ax.plot(df["t"], df[f"{v}_pos"], color="black", lw=1.8, label="reference (RK4)")
                ax.plot(df["t"], df[f"{v}_pinn"], color=self.COLORS[v], lw=1.2, ls="--", label="PINN")
                if col == 1:
                    self._windows(ax, T)
                if col == 0:
                    ax.set_ylabel(v)
                if row == 0:
                    ax.set_title(f"{method}\nrelative L2 = {self._rel_l2(df):.1e}", fontsize=10)
                ax.legend(fontsize=7, loc="upper right")
        for ax in axes[-1]:
            ax.set_xlabel("t")
        fig.suptitle(f"x, y, z over time, T = {T:g}, data at every point (dotted: window edges)")
        fig.tight_layout()
        self._save(fig, save_name or f"states_compare_T{T:g}.png")
        return fig

    # ------------------------------------------------------------------
    # 2. errors, same layout
    # ------------------------------------------------------------------

    def plot_errors_compare(self, one_pred, tm_pred, T, save_name=None):
        """|PINN - reference| for x, y, z and the 3D distance, log scale, one column per method."""
        preds = [None if p is None else self._load(p) for p in (one_pred, tm_pred)]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, df, method in zip(axes, preds, self.METHODS):
            if df is None:
                ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", color="gray")
                ax.set_title(method, fontsize=10)
                continue
            for v in self.VARS:
                ax.plot(df["t"], df[f"{v}_err"].abs() + 1e-14, color=self.COLORS[v], lw=0.9,
                        label=f"|{v} error|")
            ax.plot(df["t"], df["err_3d"] + 1e-14, color="black", lw=1.1, label="3D distance")
            ax.axhline(1e-4, color="gray", ls="--", lw=1, label="1e-4 target")
            if method.startswith("time"):
                self._windows(ax, T)
            ax.set_yscale("log")
            ax.set_ylim(1e-9, 1e2)
            ax.set_xlabel("t")
            ax.set_title(f"{method}\nrelative L2 = {self._rel_l2(df):.1e}, "
                         f"worst point = {df[['x_err', 'y_err', 'z_err']].abs().values.max():.1e}",
                         fontsize=10)
            ax.legend(fontsize=7, loc="lower right", ncol=2)
        axes[0].set_ylabel("absolute error")
        fig.suptitle(f"Error over time, T = {T:g}, data at every point")
        fig.tight_layout()
        self._save(fig, save_name or f"errors_compare_T{T:g}.png")
        return fig

    # ------------------------------------------------------------------
    # 3. which part of each unit matters most
    # ------------------------------------------------------------------

    @staticmethod
    def amplification(T=5.0, window=1.0, dt=1e-3, sigma=10.0, rho=28.0, beta=8 / 3, u0=(1.0, 1.0, 1.0)):
        """How much a small error at time s grows by the end of its window.

        Integrates the Lorenz path and its tangent (Jacobian) equations with RK4.
        For s in window [a, b], the error at b is M(s) times the error at s, with
        M(s) = Phi(b) Phi(s)^-1 and Phi the tangent matrix started at I at t = a.
        The growth factor is the largest singular value of M(s): the worst-case
        stretch of an error left uncorrected at time s.

        Returns a DataFrame with t, local time, window, growth, and the path x, y, z.
        """
        def f(u):
            x, y, z = u
            return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

        def jac(u):
            x, y, z = u
            return np.array([[-sigma, sigma, 0.0], [rho - z, -1.0, -x], [y, x, -beta]])

        def step(u, P):
            k1, K1 = f(u), jac(u) @ P
            u2 = u + dt / 2 * k1
            k2, K2 = f(u2), jac(u2) @ (P + dt / 2 * K1)
            u3 = u + dt / 2 * k2
            k3, K3 = f(u3), jac(u3) @ (P + dt / 2 * K2)
            u4 = u + dt * k3
            k4, K4 = f(u4), jac(u4) @ (P + dt * K3)
            return (u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4),
                    P + dt / 6 * (K1 + 2 * K2 + 2 * K3 + K4))

        rows = []
        u = np.array(u0, dtype=float)
        n_win = int(np.ceil(T / window - 1e-9))
        steps = int(round(window / dt))
        for k in range(n_win):
            P = np.eye(3)
            path, phis = [u.copy()], [P.copy()]
            for _ in range(steps):
                u, P = step(u, P)
                path.append(u.copy())
                phis.append(P.copy())
            end = phis[-1]
            for i, (ui, Pi) in enumerate(zip(path, phis)):
                g = np.linalg.svd(end @ np.linalg.inv(Pi), compute_uv=False)[0]
                rows.append({"t": k * window + i * dt, "local_t": i * dt, "window": k,
                             "growth": g, "x": ui[0], "y": ui[1], "z": ui[2]})
        return pd.DataFrame(rows)

    def plot_unit_importance(self, T=5.0, window=1.0, save_name=None):
        """Where in each 1-unit window data matters most.

        Left: for every window, how much an error left at each point grows by the
        end of the window (log scale, on the window's own clock). Points where this
        is large are where a wrong guess is most costly, so data there does the
        most good. Right: the same curve along the full path, with the reference x(t)
        underneath so the growth can be matched to loops and wing switches.
        """
        amp = self.amplification(T, window)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
        cmap = plt.get_cmap("viridis")
        n = amp["window"].nunique()
        for k, g in amp.groupby("window"):
            ax1.plot(g["local_t"], g["growth"], color=cmap(k / max(1, n - 1)), lw=1.4,
                     label=f"unit {k + 1}: t = {k * window:g} to {(k + 1) * window:g}")
        med = amp.groupby(amp["local_t"].round(3))["growth"].median()
        ax1.plot(med.index, med.values, color="black", lw=2.2, ls="--", label="median over units")
        ax1.set_yscale("log")
        ax1.set_xlabel("position inside the unit (0 = start, 1 = end)")
        ax1.set_ylabel("growth of an error by the unit's end")
        ax1.set_title("How much an error grows by the unit's end (chaos only)", fontsize=10)
        ax1.legend(fontsize=7)

        ax2.plot(amp["t"], amp["growth"], color="tab:purple", lw=1.3, label="error growth to unit end")
        ax2.set_yscale("log")
        ax2.set_ylabel("growth to unit end", color="tab:purple")
        ax2.set_xlabel("t")
        self._windows(ax2, T, window)
        tw = ax2.twinx()
        tw.plot(amp["t"], amp["x"], color="gray", lw=0.9, alpha=0.8, label="reference x(t)")
        tw.axhline(0, color="gray", lw=0.5)
        tw.set_ylabel("x (sign = which wing)", color="gray")
        ax2.set_title("Growth along the path stays small (about 0.8x to 2x per unit)", fontsize=10)
        fig.tight_layout()
        self._save(fig, save_name or f"unit_importance_T{T:g}.png")
        return fig, amp


def plot_unit_placement(results, save_dir=None, save_name="unit_placement.png"):
    """Measured: which part of each unit needs data.

    results : CSV path or DataFrame from unit_placement runs, one row per
              (unit, piece_center) with the window's relative L2 error. Each
              window was trained on its own from the true starting state with a
              single 0.05-long data piece at piece_center (or none).
    Left: grouped bars, one group per unit. Right: the reference path with the
    tested positions marked, so each unit's best spot can be tied to the shape.
    """
    df = pd.read_csv(results) if isinstance(results, str) else results.copy()
    df["piece_center"] = df["piece_center"].astype(str)
    order = ["none"] + sorted([p for p in df["piece_center"].unique() if p != "none"], key=float)
    labels = {"none": "no data"}
    labels.update({p: f"piece at {float(p):g} of the unit" for p in order if p != "none"})
    colors = dict(zip(order, ["lightgray", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]))
    units = sorted(df["unit"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})
    width = 0.8 / len(order)
    for i, p in enumerate(order):
        d = df[df["piece_center"] == p].set_index("unit").reindex(units)
        ax1.bar(np.arange(len(units)) + (i - (len(order) - 1) / 2) * width, d["rel_l2"], width * 0.92,
                color=colors[p], label=labels[p])
    ax1.axhline(1e-4, color="gray", ls="--", lw=1, label="1e-4 target")
    ax1.set_yscale("log")
    ax1.set_xticks(np.arange(len(units)))
    ax1.set_xticklabels([f"unit {u}\nt = {u - 1} to {u}" for u in units])
    ax1.set_ylabel("relative L2 error in that unit")
    ax1.set_title("Error in each unit with one short data piece (lower is better)", fontsize=10)
    ax1.legend(fontsize=7, ncol=2)

    T = max(units)
    amp = Horizon_Visuals.amplification(T=T)
    for v, c in zip(["x", "y", "z"], ["tab:blue", "tab:orange", "tab:green"]):
        ax2.plot(amp["t"], amp[v], color=c, lw=1.1, label=v)
    for u in units:
        for p in order:
            if p == "none":
                continue
            ax2.axvline(u - 1 + float(p), color=colors[p], lw=1.2, alpha=0.6)
    Horizon_Visuals._windows(ax2, T)
    ax2.set_xlabel("t")
    ax2.set_title("Reference path with the tested piece positions", fontsize=10)
    ax2.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, save_name), dpi=150, bbox_inches="tight")
    return fig


def plot_window_lengths(pred_dir, T=10, windows=(0.5, 1, 1.5, 2), save_dir=None,
                        save_name="window_lengths_T10.png", every=10):
    """Time-marching at one T with different window lengths: error over time,
    and the error handed from each window to the next."""
    colors = {0.5: "#2e9e6b", 1: "#2a78d6", 1.5: "#8e5cc9", 2: "#eb6834"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    for w in windows:
        f = os.path.join(pred_dir, f"pred_tm_T{T:g}_w{w:g}_s99.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        ref = np.sqrt((d[["x_pos", "y_pos", "z_pos"]] ** 2).sum(axis=1))
        rel = np.linalg.norm(d.err_3d) / np.linalg.norm(ref)
        c = colors.get(w, None)
        dd = d.iloc[::every]
        ax1.plot(dd.t, dd.err_3d + 1e-12, color=c, lw=1.1, label=f"{w:g}-unit windows (rel. L2 {rel:.1e})")
        ends = d.groupby("window").tail(1)
        ax2.plot(ends.t, ends.err_3d, marker="o", ms=4, color=c, lw=1.4, label=f"{w:g}-unit windows")
    for ax in (ax1, ax2):
        ax.set_yscale("log")
        ax.axhline(1e-4, color="gray", ls="--", lw=1)
        ax.set_xlabel("t")
        ax.set_xlim(0, T)
    ax1.set_ylim(1e-8, 1e2)
    ax1.set_ylabel("3D distance error")
    ax1.set_title(f"T = {T:g}, data everywhere: error over time for each window length", fontsize=10)
    ax1.legend(frameon=False, fontsize=8, loc="lower right")
    ax2.set_ylabel("error at the end of each window (handed to the next)")
    ax2.set_title("Hand-off error: a jump marks a window that got stuck", fontsize=10)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, save_name), dpi=150, bbox_inches="tight")
    return fig


def plot_window_states(pred_dir, T=10, windows=(0.5, 1, 1.5, 2), save_dir=None,
                       save_name="window_states_T10.png", every=10):
    """x, y, z against the reference for each window length (one column each)."""
    found = [(w, os.path.join(pred_dir, f"pred_tm_T{T:g}_w{w:g}_s99.csv")) for w in windows]
    found = [(w, f) for w, f in found if os.path.exists(f)]
    fig, axes = plt.subplots(3, len(found), figsize=(4.2 * len(found), 7.5), sharex=True, sharey="row",
                             squeeze=False)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for j, (w, f) in enumerate(found):
        d = pd.read_csv(f).iloc[::every]
        ref = np.sqrt((d[["x_pos", "y_pos", "z_pos"]] ** 2).sum(axis=1))
        full = pd.read_csv(f)
        ref_full = np.sqrt((full[["x_pos", "y_pos", "z_pos"]] ** 2).sum(axis=1))
        rel = np.linalg.norm(full.err_3d) / np.linalg.norm(ref_full)
        for i, v in enumerate("xyz"):
            ax = axes[i, j]
            ax.plot(d.t, d[f"{v}_pos"], color="black", lw=1.6, label="reference (RK4)")
            ax.plot(d.t, d[f"{v}_pinn"], color=colors[i], lw=1.2, ls="--", label="PINN")
            for k in np.arange(w, T - 1e-9, w):
                ax.axvline(k, color="gray", lw=0.5, ls=":")
            if j == 0:
                ax.set_ylabel(v)
            if i == 0:
                ax.set_title(f"{w:g}-unit windows\nrelative L2 = {rel:.1e}", fontsize=10)
            if i == 2:
                ax.set_xlabel("t")
            if j == len(found) - 1:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"x, y, z over time, T = {T:g}, time-marching with data at every point (dotted: window edges)",
                 fontsize=11)
    fig.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, save_name), dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", required=True, help="folder with pred_T*_s99.csv and pred_tm_T*_s99.csv")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--T", type=float, nargs="*", default=[1.5, 2, 2.5, 3, 5])
    a = p.parse_args()

    hv = Horizon_Visuals(save_dir=a.save_dir)
    for T in a.T:
        one = os.path.join(a.pred_dir, f"pred_T{T:g}_s99.csv")
        tm = os.path.join(a.pred_dir, f"pred_tm_T{T:g}_s99.csv")
        one, tm = (f if os.path.exists(f) else None for f in (one, tm))
        if one or tm:
            hv.plot_states_compare(one, tm, T)
            hv.plot_errors_compare(one, tm, T)
            plt.close("all")
    hv.plot_unit_importance(T=5)
    placement = os.path.join(a.pred_dir, "unit_placement.csv")
    if os.path.exists(placement):
        plot_unit_placement(placement, save_dir=a.save_dir)
    if any(f.startswith("pred_tm_T10_w") for f in os.listdir(a.pred_dir)):
        plot_window_lengths(a.pred_dir, T=10, save_dir=a.save_dir)
        plot_window_states(a.pred_dir, T=10, save_dir=a.save_dir)
    print("plots written to", a.save_dir)
