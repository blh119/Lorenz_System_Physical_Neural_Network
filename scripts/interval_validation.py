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

Or from Python:

    from interval_validation import Interval_Validation_Runs, make_intervals, sweep_configs
    runs = Interval_Validation_Runs("runs/mine")
    runs.run("three_pieces", make_intervals(3, 0.45), physics_on="full")
    runs.run_all(sweep_configs("single"))
    runs.results_df   # pandas DataFrame, also saved as runs/mine/results.csv
"""

import argparse
import os
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
    The initial condition at t = 0 is always known (hard_ic), as in the base solver.
    Everything else is passed through to Lorenz_PINN_solver / Lorenz.
    """

    def __init__(self, train_intervals=((0.0, 0.5),), physics_on="train", **solver_kwargs):
        super().__init__(**solver_kwargs)
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
               "physics_on": self.physics_on,
               "n_data": len(self.sample_list[self.dt_list.index(self.train_dt)])}
        for name, mask in [("train", train), ("held_out", ~train), ("gaps", ~train & inside_hull),
                           ("beyond", ~train & ~inside_hull), ("full", np.ones_like(train))]:
            row[f"{name}_rel_l2"], row[f"{name}_max_abs"] = self.region_errors(mask)
        row["final_loss"] = self.loss_history[-1]["total"]
        row["lbfgs_steps"] = len(self.loss_history) - self.first_stage_epochs
        return row

    # -------- plot --------

    def plot_errors(self, save_path=None, title=None):
        """|PINN - reference| over time with the training intervals shaded."""
        df = self.pred_df
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for i, (a, b) in enumerate(self.train_intervals):
            ax.axvspan(a, b, color="tab:green", alpha=0.12, label="training data" if i == 0 else None)
        for v, c in zip("xyz", ["tab:blue", "tab:orange", "tab:green"]):
            ax.semilogy(df["t"], np.maximum(df[f"{v}_err"].abs(), 1e-12), color=c, lw=1, label=f"|{v} error|")
        ax.axhline(1e-4, color="black", ls="--", lw=0.8, label="1e-4")
        ax.set_xlabel("t")
        ax.set_ylabel("absolute error (log scale)")
        ax.set_title(title or f"Train on {self.summary()['intervals']} (physics on {self.physics_on})")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return fig


# ----------------------------------------------------------------------
# Running one or many configurations
# ----------------------------------------------------------------------

def run_one(intervals, physics_on="train", epochs=TUNED_EPOCHS, lbfgs_iters=None,
            n_sample=TUNED_N_SAMPLE, seed=99, plot_path=None, print_every=5000, T=1.0):
    """Train one validation model with the tuned settings and return its summary row."""
    kwargs = dict(TUNED, T=T, seed=seed)
    if lbfgs_iters is not None:
        kwargs["lbfgs_iters"] = lbfgs_iters
    start = time.time()
    v = Lorenz_Interval_Validation(train_intervals=intervals, physics_on=physics_on, **kwargs)
    v.pinn_pipeline(dt=kwargs["dt"][0], n_sample=n_sample, epochs=epochs, print_every=print_every)
    row = v.summary()
    row.update(seed=seed, epochs=epochs, seconds=round(time.time() - start))
    if plot_path:
        v.plot_errors(save_path=plot_path)
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
    else:
        raise ValueError("kind must be 'single' or 'multi'")
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
                 save_plots=True, save_predictions=False, **run_kwargs):
        self.out_dir = out_dir
        self.csv_path = os.path.join(out_dir, csv_name)
        self.save_plots = save_plots
        self.save_predictions = save_predictions
        self.run_kwargs = run_kwargs  # epochs, lbfgs_iters, seed, T, ...
        os.makedirs(out_dir, exist_ok=True)
        self.results_df = pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else pd.DataFrame()

    def done(self, name):
        return not self.results_df.empty and name in set(self.results_df["name"])

    def run(self, name, intervals, physics_on="train", **overrides):
        """Train one model, add its row to results_df and rewrite the CSV."""
        if self.done(name):
            print(f"skipping {name}: already in {self.csv_path}")
            return self.results_df[self.results_df["name"] == name].iloc[0].to_dict()

        print(f"\n=== {name}: {intervals}, physics on {physics_on} ===")
        plot = os.path.join(self.out_dir, f"{name}.png") if self.save_plots else None
        row, solver = run_one(intervals, physics_on, plot_path=plot, **{**self.run_kwargs, **overrides})
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--intervals", help='training intervals, e.g. "0-0.3, 0.6-0.8"')
    p.add_argument("--n-intervals", type=int, help="number of equal pieces (with --total-length)")
    p.add_argument("--total-length", type=float, help="total training time across all pieces")
    p.add_argument("--start", type=float, default=0.0, help="earliest time a piece may start")
    p.add_argument("--end", type=float, default=None, help="latest time a piece may end (default T)")
    p.add_argument("--placement", default="spread", choices=["spread", "centered", "random"])
    p.add_argument("--physics-on", default="train", choices=["train", "full"])
    p.add_argument("--sweep", choices=["single", "multi"], help="run a preset sweep instead")
    p.add_argument("--only", help="with --sweep: run only configs whose name contains this text")
    p.add_argument("--shard", help='with --sweep: "i/n" runs every n-th config starting at i, to split across processes')
    p.add_argument("--T", type=float, default=1.0, help="length of the full solution [0, T]")
    p.add_argument("--epochs", type=int, default=TUNED_EPOCHS)
    p.add_argument("--lbfgs", type=int, default=None, help="L-BFGS steps (default: tuned 20000)")
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--out", default="interval_runs", help="folder for results.csv and plots")
    p.add_argument("--name", help="name for a single run's row in results.csv")
    p.add_argument("--save-predictions", action="store_true",
                   help="also save each run's full prediction table (t, reference, PINN, errors)")
    args = p.parse_args()

    run_kwargs = dict(epochs=args.epochs, lbfgs_iters=args.lbfgs, seed=args.seed, T=args.T)

    if args.sweep:
        configs = [c for c in sweep_configs(args.sweep, args.T) if not args.only or args.only in c[0]]
        if args.shard:
            i, n = map(int, args.shard.split("/"))
            configs = configs[i::n]
        runs = Interval_Validation_Runs(args.out, save_predictions=args.save_predictions, **run_kwargs)
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
        runs = Interval_Validation_Runs(args.out, save_predictions=args.save_predictions, **run_kwargs)
        row = runs.run(name, intervals, args.physics_on)
        for k, val in row.items():
            print(f"{k:>18}: {val}")
        print(f"saved to {runs.csv_path}")
