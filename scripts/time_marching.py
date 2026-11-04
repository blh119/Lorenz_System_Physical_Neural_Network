"""
Time-marching PINN for long windows [0, T].

One network over a long chaotic window gets stuck (with the tuned settings it
fails at T = 3 even with data everywhere). Time-marching splits [0, T] into
short windows and trains one network per window, in order. Each window starts
from the previous window's predicted end point, so errors carry forward the
way they would in any step-by-step solver.

Data can still be limited to chosen pieces of [0, T] (global times); each
window only sees the pieces that fall inside it. The physics is checked over
every window in full.

Example:
    from time_marching import Lorenz_Time_Marching
    tm = Lorenz_Time_Marching(T=3.0, window=1.0)                 # data everywhere
    tm = Lorenz_Time_Marching(T=3.0, data_intervals=[(0.4, 0.6), (1.4, 1.6)])
    row = tm.run()          # one-row summary; tm.pred_df has every time point
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from lorenz_pinn_full import Lorenz, DTYPE
from interval_validation import Lorenz_Interval_Validation, TUNED, TUNED_EPOCHS, parse_intervals


class Lorenz_Window(Lorenz_Interval_Validation):
    """One window of a time-marching run, in local time [0, L].

    window_df      : the true reference path for this window, time already shifted to start at 0
    data_intervals : local-time pieces with data; empty means no data in this window
    init_weights   : optional starting weights (the previous window's network)
    """

    def __init__(self, window_df, data_intervals, init_weights=None, **solver_kwargs):
        self.window_df = window_df
        self.has_data = len(data_intervals) > 0
        self.init_weights = init_weights
        intervals = data_intervals if self.has_data else [(0.0, solver_kwargs["T"])]
        super().__init__(train_intervals=intervals, physics_on="full",
                         scale_from="reference", **solver_kwargs)

    def lorenz_pipeline(self):
        # The reference for every window comes from one RK4 run over all of [0, T].
        self.lorenz_df = self.window_df

    def sample_system(self, dt=.0001, n_sample=200):
        if self.has_data:
            return super().sample_system(dt, n_sample)
        # No data here: the only "data point" is the starting state, which the
        # hard initial condition already matches, so it adds nothing.
        start = self.window_df.iloc[[0]].copy()
        start[["x_pos", "y_pos", "z_pos"]] = [self.x0, self.y0, self.z0]
        self.sample_list.append(start.reset_index(drop=True))

    def build_network(self, initial_params=None):
        super().build_network(initial_params)
        if self.init_weights is not None:
            self.model.set_weights(self.init_weights)


class Lorenz_Time_Marching:
    """Train [0, T] one window at a time with the tuned settings.

    T               : total length of the prediction
    window          : length of each window (the last one may be shorter)
    data_intervals  : global (start, end) pieces with data; None means data everywhere
    n_sample        : data points per unit of time (200 matches the tuned run)
    warm_start      : start each window's network from the previous window's weights
    stop_above      : stop early once a window's end error (3D distance) exceeds this
    """

    def __init__(self, T=3.0, window=1.0, data_intervals=None, n_sample=200, seed=99,
                 epochs=TUNED_EPOCHS, lbfgs_iters=5000, warm_start=True, print_every=5000,
                 stop_above=None):
        self.T, self.window = T, window
        self.data_intervals = [(0.0, T)] if data_intervals is None else sorted(data_intervals)
        self.n_sample, self.seed = n_sample, seed
        self.epochs, self.lbfgs_iters = epochs, lbfgs_iters
        self.warm_start, self.print_every = warm_start, print_every
        self.stop_above = stop_above  # stop once a window ends this far off (saves time on failed runs)
        self.dt = TUNED["dt"][0]
        self.window_rows = []

    def reference(self):
        ref = Lorenz(T=self.T, dt=[self.dt], method="rk4")
        ref.lorenz_pipeline()
        return ref.lorenz_df.reset_index(drop=True)

    def local_intervals(self, s, e):
        """The data pieces inside [s, e], shifted to local time."""
        out = []
        for a, b in self.data_intervals:
            a2, b2 = max(a, s), min(b, e)
            if b2 - a2 > 1e-9:
                out.append((round(a2 - s, 10), round(b2 - s, 10)))
        return out

    def run(self):
        start_time = time.time()
        ref_df = self.reference()
        n_windows = int(np.ceil(self.T / self.window - 1e-9))
        ic = (1.0, 1.0, 1.0)
        weights, preds = None, []

        for k in range(n_windows):
            s = k * self.window
            e = min(self.T, s + self.window)
            L = e - s
            wdf = ref_df[(ref_df["t"] >= s - 1e-9) & (ref_df["t"] <= e + 1e-9)].copy()
            wdf["t"] = (wdf["t"] - s).round(10)
            wdf = wdf.reset_index(drop=True)
            local = self.local_intervals(s, e)

            print(f"--- window {k + 1}/{n_windows}: [{s:g}, {e:g}], data pieces {local}")
            w = Lorenz_Window(wdf, local, init_weights=weights if self.warm_start else None,
                              T=L, x0=ic[0], y0=ic[1], z0=ic[2],
                              dt=[self.dt], method="rk4", optimizer=TUNED["optimizer"],
                              learning_rate=TUNED["learning_rate"], lbfgs_iters=self.lbfgs_iters,
                              seed=self.seed, n_colloc=int(round(2000 * L)),
                              n_colloc_lbfgs=int(round(4000 * L)))
            w.pinn_pipeline(dt=self.dt, n_sample=max(2, int(round(self.n_sample * L))),
                            epochs=self.epochs, print_every=self.print_every)

            p = w.pred_df.copy()
            p["t"] = p["t"] + s
            p["window"] = k
            preds.append(p if k == 0 else p.iloc[1:])
            last = p.iloc[-1]
            ic = (float(last["x_pinn"]), float(last["y_pinn"]), float(last["z_pinn"]))
            weights = w.model.get_weights()
            self.window_rows.append({"window": k, "start": s, "end": e,
                                     "n_data": len(w.sample_list[0]) if w.has_data else 0,
                                     "final_loss": w.loss_history[-1]["total"],
                                     "end_err_3d": float(last["err_3d"])})
            if self.stop_above is not None and float(last["err_3d"]) > self.stop_above:
                print(f"--- stopping: window {k + 1} ended {float(last['err_3d']):.2e} off (> {self.stop_above:g})")
                break

        self.pred_df = pd.concat(preds).reset_index(drop=True)
        return self.summary(time.time() - start_time)

    def in_data(self, t):
        mask = np.zeros(len(t), dtype=bool)
        for a, b in self.data_intervals:
            mask |= (t >= a) & (t <= b)
        return mask

    def region(self, mask):
        df = self.pred_df[mask]
        if len(df) == 0:
            return np.nan, np.nan
        ref_3d = np.sqrt((df[["x_pos", "y_pos", "z_pos"]] ** 2).sum(axis=1))
        return (float(np.linalg.norm(df["err_3d"]) / np.linalg.norm(ref_3d)),
                float(df[["x_err", "y_err", "z_err"]].abs().values.max()))

    def summary(self, seconds):
        t = self.pred_df["t"].values
        data = self.in_data(t)
        row = {"T": self.T, "reached_t": float(self.pred_df["t"].iloc[-1]),
               "window": self.window, "seed": self.seed,
               "intervals": "; ".join(f"{a:g}-{b:g}" for a, b in self.data_intervals),
               "data_length": round(sum(b - a for a, b in self.data_intervals), 10),
               "epochs": self.epochs, "lbfgs_cap": self.lbfgs_iters, "warm_start": self.warm_start}
        for name, mask in [("full", np.ones_like(data)), ("data", data), ("held_out", ~data)]:
            row[f"{name}_rel_l2"], row[f"{name}_max_abs"] = self.region(mask)
        for wr in self.window_rows:
            row[f"end_err3d_w{wr['window']}"] = wr["end_err_3d"]
        row["seconds"] = round(seconds)
        return row


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--T", type=float, required=True)
    p.add_argument("--window", type=float, default=1.0)
    p.add_argument("--intervals", help='global data pieces, e.g. "0.4-0.6, 1.4-1.6" (default: everywhere)')
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--epochs", type=int, default=TUNED_EPOCHS)
    p.add_argument("--lbfgs", type=int, default=5000)
    p.add_argument("--no-warm-start", action="store_true")
    p.add_argument("--stop-above", type=float, help="stop once a window ends this far off the reference")
    p.add_argument("--name", default="", help="label stored in the CSV row")
    p.add_argument("--csv", help="append the summary row to this CSV")
    p.add_argument("--save-predictions", help="write the full prediction table here")
    a = p.parse_args()

    tm = Lorenz_Time_Marching(T=a.T, window=a.window, seed=a.seed, epochs=a.epochs,
                              lbfgs_iters=a.lbfgs, warm_start=not a.no_warm_start, stop_above=a.stop_above,
                              data_intervals=parse_intervals(a.intervals) if a.intervals else None)
    row = {"name": a.name, **tm.run()}
    print("ROW", json.dumps(row))
    if a.csv:
        # Rewrite instead of appending: runs with more windows add columns.
        df = pd.DataFrame([row])
        if os.path.exists(a.csv):
            df = pd.concat([pd.read_csv(a.csv), df], ignore_index=True)
        df.to_csv(a.csv, index=False)
    if a.save_predictions:
        tm.pred_df.to_csv(a.save_predictions, index=False)
