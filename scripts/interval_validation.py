"""Train-interval validation for the Lorenz PINN.

Train the PINN on one or more pieces of [0, T] and check how well it predicts
the parts it never saw data for.

Quick start (from the scripts/ folder):

    # one run, data on [0, 0.3] and [0.6, 0.8], physics only on those pieces
    python interval_validation.py --intervals "0-0.3, 0.6-0.8" --physics-on train

    # same intervals, but physics on all of [0, 1] (data still only on the pieces)
    python interval_validation.py --intervals "0-0.3, 0.6-0.8" --physics-on full

    # 3 equal pieces adding up to 0.45, spread evenly across [0, 1]
    python interval_validation.py --n-intervals 3 --total-length 0.45

    # every run is a row in <out>/results.csv (re-running skips finished runs)
    # a whole sweep:
    python interval_validation.py --sweep single --out runs/single
    python interval_validation.py --sweep multi --out runs/multi

    # sweep plots (heatmaps, error vs length, error vs number of intervals)
    python interval_validation.py --plot runs/single/results.csv runs/multi/results.csv --out runs/plots

Or from Python:

    from interval_validation import Interval_Validation_Runs, make_intervals, sweep_configs
    runs = Interval_Validation_Runs("runs/mine")
    runs.run("three_pieces", make_intervals(3, 0.45), physics_on="full")
    runs.run_all(sweep_configs("single"))
    runs.results_df   # pandas DataFrame, also saved as runs/mine/results.csv
    Interval_Validation_Visuals(solver=runs.last_solver, save_dir="plots").plot_run_all()
    Interval_Validation_Visuals(results_df=runs.results_df, save_dir="plots").plot_sweep_all()
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from lorenz_pinn_full import Lorenz_PINN_solver, DTYPE


# Tuned settings from results/all_results.csv. Every validation run uses these.
TUNED = dict(dt=[.0001], method="rk4", optimizer="adam", learning_rate=1e-2,
             lbfgs_iters=20000)
TUNED_EPOCHS = 10000
TUNED_N_SAMPLE = 200  # data points per unit of time (200 on [0, 1] in the tuned run)


# ----------------------------------------------------------------------
# Ways to write down the training intervals
# ----------------------------------------------------------------------

def parse_intervals(text):
    """"0-0.3, 0.6-0.8" -> [(0.0, 0.3), (0.6, 0.8)]"""
    intervals = []
    for piece in text.split(","):
        a, b = piece.strip().split("-")
        intervals.append((float(a), float(b)))
    return intervals


def make_intervals(n, total_length, start=0.0, end=1.0, placement="spread", seed=0):
    """n equal pieces whose lengths add up to total_length, inside [start, end].

    placement : "spread"  first piece starts at `start`, last ends at `end`,
                          equal gaps in between (n = 1 sits at `start`)
                "centered" equal gaps before, between and after the pieces
                "random"   random non-overlapping positions (uses seed)
    """
    span = end - start
    if not 0 < total_length <= span:
        raise ValueError("total_length must be > 0 and fit inside [start, end]")
    piece = total_length / n
    free = span - total_length

    if placement == "spread":
        gap = free / (n - 1) if n > 1 else 0.0
        starts = [start + i * (piece + gap) for i in range(n)]
    elif placement == "centered":
        gap = free / (n + 1)
        starts = [start + gap + i * (piece + gap) for i in range(n)]
    elif placement == "random":
        # split the free time into n + 1 random gaps
        gaps = np.random.default_rng(seed).dirichlet(np.ones(n + 1)) * free
        starts = [start + gaps[:i + 1].sum() + i * piece for i in range(n)]
    else:
        raise ValueError("placement must be 'spread', 'centered' or 'random'")

    return [(round(s, 10), round(s + piece, 10)) for s in starts]


def clean_intervals(intervals, T):
    """Sort, clip to [0, T] and merge overlaps, so lengths add up correctly."""
    out = []
    for a, b in sorted((max(0.0, a), min(T, b)) for a, b in intervals):
        if b <= a:
            raise ValueError(f"interval ({a}, {b}) is empty inside [0, {T}]")
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


# ----------------------------------------------------------------------
# The validation solver
# ----------------------------------------------------------------------

class Lorenz_Interval_Validation(Lorenz_PINN_solver):
    """Lorenz_PINN_solver that only sees data on chosen pieces of [0, T].

    Everything about the network and training is inherited unchanged. Only
    two things move: where the data points come from, and where the physics
    (collocation) points go.

    train_intervals : list of (start, end) pairs, e.g. [(0, 0.3), (0.6, 0.8)].
                      Overlaps are merged; total length must be at most T.
    physics_on      : "train" physics points only inside the intervals, so the
                              network gets no information at all outside them
                              (strict extrapolation)
                      "full"  physics points on all of [0, T]; data still only
                              inside the intervals. The ODE itself is free to
                              enforce anywhere, so this tests how far the
                              physics can carry the solution past the data.
    scale_from      : where the output scaling (mean and spread of x, y, z) comes from.
                      "reference" (default) uses the whole reference path on [0, T],
                      the same three numbers the tuned model used. "train" uses only
                      the training samples; on a short interval the spread is tiny,
                      which blows up the scaled residual and stalls training, so
                      that option mostly measures a scaling problem, not extrapolation.
    The initial condition at t = 0 is always known (hard_ic), as in the base solver.
    Everything else is passed through to Lorenz_PINN_solver / Lorenz.
    """

    def __init__(self, train_intervals=((0.0, 0.5),), physics_on="train", scale_from="reference",
                 **solver_kwargs):
        super().__init__(**solver_kwargs)
        # The base class only calls tf.random.set_seed, which doesn't pin the
        # Keras weight initializer: the starting weights then depend on how
        # many models the process built before. This pins Python, NumPy, TF
        # and Keras together, so a run's seed fully decides its starting point.
        tf.keras.utils.set_random_seed(self.seed)
        if scale_from not in ("reference", "train"):
            raise ValueError("scale_from must be 'reference' or 'train'")
        self.scale_from = scale_from
        if physics_on not in ("train", "full"):
            raise ValueError("physics_on must be 'train' or 'full'")
        self.physics_on = physics_on
        self.train_intervals = clean_intervals(train_intervals, self.T)
        self.lengths = [b - a for a, b in self.train_intervals]
        self.train_length = sum(self.lengths)

        # Physics points split across intervals in proportion to their length,
        # so the density matches the base solver's n_colloc per T.
        self.colloc_counts = self._split(self.n_colloc)
        self.colloc_counts_lbfgs = self._split(self.n_colloc_lbfgs)

    def _split(self, n_total):
        """Share n_total points (per T) across the intervals by length."""
        return [max(2, int(round(n_total * L / self.T))) for L in self.lengths]

    def in_train(self, t):
        """Boolean mask: which times fall inside a training interval."""
        t = np.asarray(t)
        mask = np.zeros(t.shape, dtype=bool)
        for a, b in self.train_intervals:
            mask |= (t >= a) & (t <= b)
        return mask

    def prepare_training_data(self, dt=.0001):
        super().prepare_training_data(dt)
        if self.scale_from == "reference":
            ref = self.lorenz_df.loc[self.lorenz_df["dt"] == dt, ["x_pos", "y_pos", "z_pos"]]
            self.state_mean = tf.constant(ref.mean().values.reshape(1, 3), dtype=DTYPE)
            self.state_std = tf.constant(ref.std().values.reshape(1, 3), dtype=DTYPE)

    # -------- data: stratified sample inside each interval --------

    def sample_system(self, dt=.0001, n_sample=TUNED_N_SAMPLE):
        """n_sample is points per T, so a 0.3-long interval gets 0.3 * n_sample."""
        current_df = self.lorenz_df.loc[self.lorenz_df["dt"] == dt, ].reset_index(drop=True)

        pieces = []
        for (a, b), L in zip(self.train_intervals, self.lengths):
            part = current_df[(current_df["t"] >= a) & (current_df["t"] <= b)]
            n = max(2, int(round(n_sample * L / self.T)))
            bins = pd.cut(part["t"], bins=n)
            pieces.append(part.groupby(bins, observed=True).sample(n=1, random_state=self.seed))

        self.sample_list.append(pd.concat(pieces).sort_values("t").reset_index(drop=True))

    # -------- physics points --------

    def random_colloc(self):
        if self.physics_on == "full":
            return super().random_colloc()
        return tf.concat([tf.random.uniform((n, 1), a, b, dtype=DTYPE)
                          for (a, b), n in zip(self.train_intervals, self.colloc_counts)], axis=0)

    def fixed_colloc(self):
        if self.physics_on == "full":
            return super().fixed_colloc()
        pts = np.concatenate([np.linspace(a, b, n) for (a, b), n
                              in zip(self.train_intervals, self.colloc_counts_lbfgs)])
        return tf.constant(pts.reshape(-1, 1), dtype=DTYPE)

    # -------- scoring --------

    def region_errors(self, mask):
        """Relative L2 (x, y, z combined) and worst single-coordinate error on rows in mask."""
        df = self.pred_df[mask]
        if len(df) == 0:
            return np.nan, np.nan
        ref_3d = np.sqrt((df[["x_pos", "y_pos", "z_pos"]] ** 2).sum(axis=1))
        rel_l2 = np.linalg.norm(df["err_3d"]) / np.linalg.norm(ref_3d)
        max_abs = df[["x_err", "y_err", "z_err"]].abs().values.max()
        return rel_l2, max_abs

    def summary(self):
        """One row of results: the setup plus errors on each region of [0, T].

        held_out = every time not in a training interval. It is split into
        gaps (between the first and last interval, i.e. interpolation) and
        beyond (before the first or after the last, i.e. extrapolation).
        """
        t = self.pred_df["t"].values
        train = self.in_train(t)
        first, last = self.train_intervals[0][0], self.train_intervals[-1][1]
        inside_hull = (t >= first) & (t <= last)

        row = {"intervals": "; ".join(f"{a:g}-{b:g}" for a, b in self.train_intervals),
               "n_intervals": len(self.train_intervals),
               "train_length": round(self.train_length, 10),
               "first_start": first, "last_end": last,
               "physics_on": self.physics_on, "scale_from": self.scale_from,
               "n_data": len(self.sample_list[self.dt_list.index(self.train_dt)])}
        for name, mask in [("train", train), ("held_out", ~train), ("gaps", ~train & inside_hull),
                           ("beyond", ~train & ~inside_hull), ("full", np.ones_like(train))]:
            row[f"{name}_rel_l2"], row[f"{name}_max_abs"] = self.region_errors(mask)
        row["final_loss"] = self.loss_history[-1]["total"]
        row["lbfgs_steps"] = len(self.loss_history) - self.first_stage_epochs
        return row


# ----------------------------------------------------------------------
# Running one or many configurations
# ----------------------------------------------------------------------

def run_one(intervals, physics_on="train", epochs=TUNED_EPOCHS, lbfgs_iters=None,
            n_sample=TUNED_N_SAMPLE, seed=99, plot_path=None, print_every=5000, T=1.0,
            scale_from="reference"):
    """Train one validation model with the tuned settings and return its summary row."""
    kwargs = dict(TUNED, T=T, seed=seed)
    if lbfgs_iters is not None:
        kwargs["lbfgs_iters"] = lbfgs_iters
    start = time.time()
    v = Lorenz_Interval_Validation(train_intervals=intervals, physics_on=physics_on,
                                   scale_from=scale_from, **kwargs)
    v.pinn_pipeline(dt=kwargs["dt"][0], n_sample=n_sample, epochs=epochs, print_every=print_every)
    row = v.summary()
    row.update(seed=seed, epochs=epochs, seconds=round(time.time() - start))
    if plot_path:
        Interval_Validation_Visuals(solver=v).plot_errors(save_path=plot_path)
        plt.close("all")
    return row, v


def sweep_configs(kind, T=1.0):
    """Named lists of (name, intervals, physics_on) to run."""
    configs = []
    if kind == "single":
        # one interval: length x location (start time)
        for L in [0.1, 0.25, 0.5, 0.75]:
            for a in [0.0, 0.25, 0.5, 0.75]:
                if a + L <= T + 1e-12:
                    for mode in ["train", "full"]:
                        configs.append((f"single_L{L:g}_a{a:g}_{mode}", [(a, a + L)], mode))
    elif kind == "multi":
        # n pieces with the same total length, spread across [0, T] or packed at the start
        for total in [0.25, 0.5]:
            for n in [2, 3, 5]:
                for placement in ["spread", "front"]:
                    end = T if placement == "spread" else min(T, 1.5 * total)
                    iv = make_intervals(n, total, 0.0, end, "spread")
                    for mode in ["train", "full"]:
                        configs.append((f"multi_tot{total:g}_n{n}_{placement}_{mode}", iv, mode))
    elif kind == "budget":
        # Search for the least data that still predicts the rest of [0, T].
        # physics on all of [0, T]: small total lengths, one to three pieces
        for total in [0.05, 0.1]:
            configs.append((f"budget_tot{total:g}_n1_at0.3_full", [(0.3 - total / 2, 0.3 + total / 2)], "full"))
            for n in [2, 3]:
                for placement in ["spread", "centered"]:
                    configs.append((f"budget_tot{total:g}_n{n}_{placement}_full",
                                    make_intervals(n, total, 0.0, T, placement), "full"))
        # physics only where there is data: many short pieces covering [0, T],
        # so the gaps between them stay small (interpolation, not extrapolation)
        for total in [0.25, 0.5]:
            for n in [10, 20]:
                configs.append((f"budget_tot{total:g}_n{n}_spread_train", make_intervals(n, total, 0.0, T), "train"))
    else:
        raise ValueError("kind must be 'single', 'multi' or 'budget'")
    return configs


class Interval_Validation_Runs:
    """Keeps every validation run in one DataFrame and writes it to a CSV.

        runs = Interval_Validation_Runs("runs/my_sweep")
        runs.run("first_half", [(0, 0.5)], physics_on="full")
        runs.run_all(sweep_configs("single"))
        runs.results_df            # one row per run
        runs.to_csv()              # also written after every run

    If csv_path already exists its rows are loaded first, and a run whose
    name is already there is skipped, so an interrupted sweep picks up
    where it stopped.
    """

    def __init__(self, out_dir="interval_runs", csv_name="results.csv",
                 save_plots=True, save_predictions=False, claim=False, **run_kwargs):
        self.out_dir = out_dir
        self.csv_path = os.path.join(out_dir, csv_name)
        self.save_plots = save_plots
        self.save_predictions = save_predictions
        self.run_kwargs = run_kwargs  # epochs, lbfgs_iters, seed, T, ...
        # claim=True: several processes can share one out_dir (each with its own
        # csv_name). A run is skipped if another process already left a
        # <name>.claim file; delete those files to re-run.
        self.claim = claim
        os.makedirs(out_dir, exist_ok=True)
        self.results_df = pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else pd.DataFrame()
        self.last_solver = None  # the most recently trained model, for plotting

    def done(self, name):
        return not self.results_df.empty and name in set(self.results_df["name"])

    def run(self, name, intervals, physics_on="train", overwrite=False, **overrides):
        """Train one model, add its row to results_df and rewrite the CSV.

        overwrite=True retrains a run whose name is already in the CSV and
        replaces its row, instead of skipping it.
        """
        if overwrite and self.done(name):
            self.results_df = self.results_df[self.results_df["name"] != name].reset_index(drop=True)
        if self.done(name):
            print(f"skipping {name}: already in {self.csv_path}")
            return self.results_df[self.results_df["name"] == name].iloc[0].to_dict()

        if self.claim:
            try:
                os.close(os.open(os.path.join(self.out_dir, f"{name}.claim"), os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                return None

        print(f"\n=== {name}: {intervals}, physics on {physics_on} ===")
        plot = os.path.join(self.out_dir, f"{name}.png") if self.save_plots else None
        row, solver = run_one(intervals, physics_on, plot_path=plot, **{**self.run_kwargs, **overrides})
        self.last_solver = solver
        if self.save_predictions:
            solver.pred_df.to_csv(os.path.join(self.out_dir, f"{name}_predictions.csv"), index=False)

        row = {"name": name, **row}
        self.results_df = pd.concat([self.results_df, pd.DataFrame([row])], ignore_index=True)
        self.to_csv()
        print({k: row[k] for k in ["held_out_rel_l2", "held_out_max_abs", "train_rel_l2", "seconds"]})
        return row

    def run_all(self, configs):
        """configs: list of (name, intervals, physics_on), e.g. from sweep_configs."""
        for name, intervals, physics_on in configs:
            self.run(name, intervals, physics_on)
        return self.results_df

    def to_csv(self, path=None):
        self.results_df.to_csv(path or self.csv_path, index=False)


class Interval_Validation_Visuals:
    """Plots for train-interval validation.

    Two kinds of input, either or both:
        solver     : a trained Lorenz_Interval_Validation (after predict), for
                     plots of one run: where it trained and how the error
                     behaves inside and outside those intervals
        results_df : the table of many runs (Interval_Validation_Runs.results_df,
                     or results.csv files via from_csv), for sweep plots

        vis = Interval_Validation_Visuals(solver=v, save_dir="plots")
        vis.plot_run_all()
        vis = Interval_Validation_Visuals.from_csv("runs/single/results.csv", "runs/multi/results.csv")
        vis.plot_location_length_heatmap(physics_on="full")

    Every plot method returns its figure and saves it if you pass save_path.
    """

    VARS = ["x", "y", "z"]
    COLORS = {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}
    MODE_COLORS = {"train": "tab:orange", "full": "tab:blue"}
    MODE_LABELS = {"train": "physics on training intervals only", "full": "physics on all of [0, T]"}
    TRAIN_SHADE = dict(color="tab:green", alpha=0.12)
    METRIC_LABELS = {"held_out_rel_l2": "held-out relative L2 error",
                     "held_out_max_abs": "held-out worst-point error",
                     "full_rel_l2": "relative L2 error on all of [0, T]",
                     "full_max_abs": "worst-point error on all of [0, T]",
                     "beyond_rel_l2": "relative L2 error beyond the intervals",
                     "gaps_rel_l2": "relative L2 error in the gaps",
                     "train_rel_l2": "relative L2 error on the training intervals"}

    def __init__(self, solver=None, results_df=None, save_dir=None):
        if solver is not None and not hasattr(solver, "pred_df"):
            raise ValueError("Run solver.predict() (or pinn_pipeline) before making visuals.")
        self.solver = solver
        self.results_df = results_df
        self.save_dir = save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    @classmethod
    def from_csv(cls, *paths, save_dir=None):
        """Load and stack one or more results.csv files."""
        return cls(results_df=pd.concat([pd.read_csv(p) for p in paths], ignore_index=True),
                   save_dir=save_dir)

    # -------- helpers --------

    def _save(self, fig, save_path):
        if save_path is None:
            return
        if self.save_dir is not None and not os.path.isabs(save_path):
            save_path = os.path.join(self.save_dir, save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    def _shade(self, ax, label=True):
        for i, (a, b) in enumerate(self.solver.train_intervals):
            ax.axvspan(a, b, **self.TRAIN_SHADE, label="training data" if label and i == 0 else None)

    def _need(self, what):
        if what == "solver" and self.solver is None:
            raise ValueError("This plot needs a trained solver: Interval_Validation_Visuals(solver=...)")
        if what == "results" and self.results_df is None:
            raise ValueError("This plot needs results: Interval_Validation_Visuals(results_df=...) or from_csv")

    def _title(self):
        s = self.solver
        iv = ", ".join(f"[{a:.3g}, {b:.3g}]" for a, b in s.train_intervals)
        return f"Trained on {iv} ({self.MODE_LABELS[s.physics_on]})"

    # -------- one run --------

    def plot_errors(self, save_path=None):
        """|PINN - reference| over time, log scale, training intervals shaded."""
        self._need("solver")
        df = self.solver.pred_df
        fig, ax = plt.subplots(figsize=(10, 4.5))
        self._shade(ax)
        for v in self.VARS:
            ax.semilogy(df["t"], np.maximum(df[f"{v}_err"].abs(), 1e-12), color=self.COLORS[v],
                        lw=1, label=f"|{v} error|")
        ax.axhline(1e-4, color="black", ls="--", lw=0.8, label="1e-4 target")
        ax.set_xlabel("t")
        ax.set_ylabel("absolute error (log scale)")
        ax.set_title(self._title(), fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_time_series(self, save_path=None):
        """x(t), y(t), z(t): PINN vs reference, training intervals shaded."""
        self._need("solver")
        df, s = self.solver.pred_df, self.solver
        samples = s.sample_list[s.dt_list.index(s.train_dt)]
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for ax, v in zip(axes, self.VARS):
            self._shade(ax, label=(v == "x"))
            ax.plot(df["t"], df[f"{v}_pos"], color="black", lw=1.5, label="reference")
            ax.plot(df["t"], df[f"{v}_pinn"], color=self.COLORS[v], lw=1.5, ls="--", label="PINN")
            ax.scatter(samples["t"], samples[f"{v}_pos"], color="tab:red", s=8, zorder=3,
                       label="training samples" if v == "x" else None)
            ax.set_ylabel(v)
            ax.grid(alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].set_title(self._title(), fontsize=10)
        axes[-1].set_xlabel("t")
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_3d(self, save_path=None):
        """Reference path in gray; PINN path in green where it had data, red where it did not."""
        self._need("solver")
        df = self.solver.pred_df
        train = self.solver.in_train(df["t"].values)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(projection="3d")
        ax.plot(df["x_pos"], df["y_pos"], df["z_pos"], color="gray", lw=1, alpha=0.6, label="reference")
        for mask, color, label in [(train, "tab:green", "PINN, training intervals"),
                                   (~train, "tab:red", "PINN, held out")]:
            # NaN outside the mask breaks the line there instead of joining across gaps
            part = df[["x_pinn", "y_pinn", "z_pinn"]].where(pd.Series(mask, index=df.index), axis=0)
            ax.plot(part["x_pinn"], part["y_pinn"], part["z_pinn"], color=color, lw=1.4, label=label)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(self._title(), fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
        self._save(fig, save_path)
        return fig

    def plot_run_all(self, show=False):
        """Every one-run plot (saved too if save_dir was given)."""
        save = self.save_dir is not None
        figs = {"errors": self.plot_errors("errors.png" if save else None),
                "time_series": self.plot_time_series("time_series.png" if save else None),
                "trajectory_3d": self.plot_3d("trajectory_3d.png" if save else None)}
        if show:
            plt.show()
        return figs

    # -------- many runs --------

    def plot_location_length_heatmap(self, metric="held_out_rel_l2", physics_on="full", save_path=None):
        """Single-interval runs: interval start (columns) x length (rows), colored by error.

        Darker = larger error, log color scale. Each cell is labeled with its value.
        """
        self._need("results")
        from matplotlib.colors import LogNorm
        df = self.results_df[(self.results_df["n_intervals"] == 1) & (self.results_df["physics_on"] == physics_on)]
        # several runs with the same start and length (e.g. other seeds): show the first, don't average
        grid = df.pivot_table(index="train_length", columns="first_start", values=metric, aggfunc="first")
        fig, ax = plt.subplots(figsize=(7, 4.8))
        vals = grid.values
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("#eeeeee")  # combinations that don't fit in [0, T] or weren't run
        im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, norm=LogNorm(np.nanmin(vals), np.nanmax(vals)),
                       origin="lower", aspect="auto")
        ax.set_xticks(range(len(grid.columns)), [f"{c:g}" for c in grid.columns])
        ax.set_yticks(range(len(grid.index)), [f"{r:g}" for r in grid.index])
        mid = np.sqrt(np.nanmin(vals) * np.nanmax(vals))
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if not np.isfinite(vals[i, j]):
                    ax.text(j, i, "not run", ha="center", va="center", fontsize=8, color="gray")
                else:
                    ax.text(j, i, f"{vals[i, j]:.1e}", ha="center", va="center", fontsize=9,
                            color="white" if vals[i, j] > mid else "black")
        ax.set_xlabel("interval start time")
        ax.set_ylabel("interval length")
        ax.set_title(f"{self.METRIC_LABELS.get(metric, metric)}\n{self.MODE_LABELS[physics_on]}", fontsize=10)
        fig.colorbar(im, ax=ax, label=metric)
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_error_vs_length(self, metric="held_out_rel_l2", save_path=None):
        """Every run: error against total training length. Circles = one interval, squares = several."""
        self._need("results")
        df = self.results_df
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for mode, color in self.MODE_COLORS.items():
            for single, marker in [(True, "o"), (False, "s")]:
                part = df[(df["physics_on"] == mode) & ((df["n_intervals"] == 1) == single)]
                if len(part):
                    ax.scatter(part["train_length"], part[metric], color=color, marker=marker, s=40,
                               edgecolor="white", linewidth=1, zorder=3,
                               label=f"{self.MODE_LABELS[mode]}, {'one interval' if single else 'several'}")
        ax.set_yscale("log")
        ax.axhline(1e-4, color="black", ls="--", lw=0.8, label="1e-4 target")
        ax.set_xlabel("total training length (out of T)")
        ax.set_ylabel(self.METRIC_LABELS.get(metric, metric) + " (log)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    PLACEMENTS = ["spread over [0, T]", "packed from t = 0", "away from both ends"]

    def _placement(self, df):
        """Name each layout by its shape: touches both ends, starts at 0 only, or neither."""
        T = df["last_end"].max()
        at0 = np.isclose(df["first_start"], 0)
        atT = np.isclose(df["last_end"], T)
        return np.select([at0 & atT, at0], self.PLACEMENTS[:2], self.PLACEMENTS[2])

    def plot_interval_count(self, metric="held_out_rel_l2", physics_on="full", save_path=None):
        """Error against the number of intervals: one panel per placement, one line per total length.

        A single interval that starts at t = 0 is drawn as n = 1 in both the
        "spread" and "packed" panels, since it is the same layout for both.
        Single intervals away from both ends are left out: there are several
        per length (one per start time), so they have no one n = 1 value.
        """
        self._need("results")
        df = self.results_df[self.results_df["physics_on"] == physics_on].copy()
        df["placement"] = self._placement(df)
        df["total"] = df["train_length"].round(6)
        totals = sorted(df.loc[df["n_intervals"] > 1, "total"].unique())
        colors = dict(zip(totals, ["tab:blue", "tab:orange", "tab:green", "tab:red",
                                   "tab:purple", "tab:brown", "tab:pink", "tab:olive"]))

        placements = [p for p in self.PLACEMENTS
                      if ((df["placement"] == p) & (df["n_intervals"] > 1)).any()]
        fig, axes = plt.subplots(1, len(placements), figsize=(4.4 * len(placements), 4.5),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        for ax, placement in zip(axes, placements):
            xs = set()
            for total in totals:
                part = df[(df["total"] == total) & (df["n_intervals"] > 1) & (df["placement"] == placement)]
                if part.empty:
                    continue
                if placement != self.PLACEMENTS[2]:
                    one = df[(df["total"] == total) & (df["n_intervals"] == 1) & np.isclose(df["first_start"], 0)]
                    part = pd.concat([one, part])
                part = part.sort_values("n_intervals")
                xs.update(part["n_intervals"])
                ax.plot(part["n_intervals"], part[metric], lw=2, marker="o", ms=8,
                        markeredgecolor="white", color=colors[total], label=f"total {total:g}")
            ax.axhline(1e-4, color="black", ls=":", lw=0.8, label="1e-4 target")
            ax.set_xscale("log")
            ax.set_xticks(sorted(xs), [str(int(n)) for n in sorted(xs)])
            ax.minorticks_off()
            ax.set_yscale("log")
            ax.set_title(placement, fontsize=10)
            ax.set_xlabel("number of training intervals")
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=8)
        axes[0].set_ylabel(self.METRIC_LABELS.get(metric, metric) + " (log)")
        fig.suptitle(self.MODE_LABELS[physics_on], fontsize=11)
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    def plot_sweep_all(self, metric="held_out_rel_l2", show=False):
        """Every sweep plot for both physics modes (saved too if save_dir was given)."""
        save = self.save_dir is not None
        figs = {"error_vs_length": self.plot_error_vs_length(metric, "error_vs_length.png" if save else None)}
        for mode in self.results_df["physics_on"].unique():
            if ((self.results_df["physics_on"] == mode) & (self.results_df["n_intervals"] == 1)).any():
                figs[f"heatmap_{mode}"] = self.plot_location_length_heatmap(
                    metric, mode, f"heatmap_{mode}.png" if save else None)
            if ((self.results_df["physics_on"] == mode) & (self.results_df["n_intervals"] > 1)).any():
                figs[f"interval_count_{mode}"] = self.plot_interval_count(
                    metric, mode, f"interval_count_{mode}.png" if save else None)
        if show:
            plt.show()
        return figs


def main_cli():
    """Command-line entry point (see the examples at the top of the file)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--intervals", help='training intervals, e.g. "0-0.3, 0.6-0.8"')
    p.add_argument("--n-intervals", type=int, help="number of equal pieces (with --total-length)")
    p.add_argument("--total-length", type=float, help="total training time across all pieces")
    p.add_argument("--start", type=float, default=0.0, help="earliest time a piece may start")
    p.add_argument("--end", type=float, default=None, help="latest time a piece may end (default T)")
    p.add_argument("--placement", default="spread", choices=["spread", "centered", "random"])
    p.add_argument("--physics-on", default="train", choices=["train", "full"])
    p.add_argument("--sweep", choices=["single", "multi", "budget"], help="run a preset sweep instead")
    p.add_argument("--only", help="with --sweep: run only configs whose name contains this text")
    p.add_argument("--T", type=float, default=1.0, help="length of the full solution [0, T]")
    p.add_argument("--epochs", type=int, default=TUNED_EPOCHS)
    p.add_argument("--lbfgs", type=int, default=None, help="L-BFGS steps (default: tuned 20000)")
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--scale-from", default="reference", choices=["reference", "train"],
                   help="output scaling from the whole reference path (default) or the training samples only")
    p.add_argument("--csv-name", default="results.csv")
    p.add_argument("--claim", action="store_true",
                   help="with --sweep: let several processes share --out (give each its own --csv-name)")
    p.add_argument("--out", default="interval_runs", help="folder for results.csv and plots")
    p.add_argument("--name", help="name for a single run's row in results.csv")
    p.add_argument("--plot", nargs="+", metavar="CSV",
                   help="only make the sweep plots from these results.csv files (into --out)")
    p.add_argument("--save-predictions", action="store_true",
                   help="also save each run's full prediction table (t, reference, PINN, errors)")
    args = p.parse_args()

    run_kwargs = dict(epochs=args.epochs, lbfgs_iters=args.lbfgs, seed=args.seed, T=args.T,
                      scale_from=args.scale_from)

    if args.plot:
        Interval_Validation_Visuals.from_csv(*args.plot, save_dir=args.out).plot_sweep_all()
        print(f"plots saved to {args.out}")
    elif args.sweep:
        configs = [c for c in sweep_configs(args.sweep, args.T) if not args.only or args.only in c[0]]
        runs = Interval_Validation_Runs(args.out, args.csv_name, save_predictions=args.save_predictions,
                                        claim=args.claim, **run_kwargs)
        print(runs.run_all(configs).to_string())
    else:
        if args.intervals:
            intervals = parse_intervals(args.intervals)
        elif args.n_intervals and args.total_length:
            intervals = make_intervals(args.n_intervals, args.total_length, args.start,
                                       args.end if args.end is not None else args.T, args.placement, args.seed)
        else:
            p.error("give --intervals, or --n-intervals with --total-length, or --sweep")
        name = args.name or "run_" + "_".join(f"{a:g}-{b:g}" for a, b in intervals) + f"_{args.physics_on}"
        runs = Interval_Validation_Runs(args.out, args.csv_name, save_predictions=args.save_predictions,
                                        **run_kwargs)
        row = runs.run(name, intervals, args.physics_on)
        if runs.last_solver is not None:
            Interval_Validation_Visuals(solver=runs.last_solver,
                                        save_dir=os.path.join(args.out, name)).plot_run_all()
        for k, val in row.items():
            print(f"{k:>18}: {val}")
        print(f"saved to {runs.csv_path}")


def main_ide(what, intervals, physics_on, lbfgs_iters, sweep, plot_csvs, out_dir, show_plots):
    """Entry point for the settings block below, for pressing Run in an IDE."""
    if what == "plot":
        vis = Interval_Validation_Visuals.from_csv(*plot_csvs, save_dir=out_dir)
        vis.plot_sweep_all(show=show_plots)
        print(f"plots saved to {out_dir}")
        return vis

    runs = Interval_Validation_Runs(out_dir, lbfgs_iters=lbfgs_iters)
    if what == "sweep":
        runs.run_all(sweep_configs(sweep))
        Interval_Validation_Visuals(results_df=runs.results_df,
                                    save_dir=os.path.join(out_dir, "plots")).plot_sweep_all(show=show_plots)
    elif what == "single":
        name = "run_" + "_".join(f"{a:.3g}-{b:.3g}" for a, b in intervals) + f"_{physics_on}"
        row = runs.run(name, intervals, physics_on, overwrite=True)
        Interval_Validation_Visuals(solver=runs.last_solver,
                                    save_dir=os.path.join(out_dir, name)).plot_run_all(show=show_plots)
        for k, val in row.items():
            print(f"{k:>18}: {val}")
    else:
        raise ValueError("what must be 'single', 'sweep' or 'plot'")
    print(f"results saved to {runs.csv_path}")
    return runs


if __name__ == "__main__":

    # ---------------- Settings for running from an IDE ----------------
    # Edit these and press Run. Running from a terminal with arguments
    # (python interval_validation.py --intervals ...) ignores this block.
    HERE = os.path.dirname(os.path.abspath(__file__))  # results go next to this file, wherever the IDE runs from

    WHAT = "single"             # "single": one run, "sweep": a preset sweep, "plot": graphs from results CSVs
    INTERVALS = make_intervals(3, 0.05, placement="centered")  # or a list, e.g. [(0, 0.3), (0.6, 0.8)]
    PHYSICS_ON = "full"         # "full": physics on all of [0, T]; "train": only inside the intervals
    LBFGS_ITERS = 20000         # tuned value (about 30 min per run); 2000 gives a quick 5-10 min run
    SWEEP = "single"            # with WHAT = "sweep": "single", "multi" or "budget"
    PLOT_CSVS = [os.path.join(HERE, "sweep_results.csv")]  # with WHAT = "plot"
    OUT_DIR = os.path.join(HERE, "interval_runs")
    SHOW_PLOTS = True           # also open the figures (saved as PNGs either way)

    if len(sys.argv) > 1:
        main_cli()
    else:
        main_ide(WHAT, INTERVALS, PHYSICS_ON, LBFGS_ITERS, SWEEP, PLOT_CSVS, OUT_DIR, SHOW_PLOTS)
