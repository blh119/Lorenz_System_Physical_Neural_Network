"""How much data does one network need on [0, T]? (Report Section 18.)

One network on all of [0, T], tuned settings, data spread evenly. Only the
amount of data (as a % of the RK4 reference points), the L-BFGS cap and the
seed change between runs. Each finished run is appended as one row to the CSV,
and a run already in the CSV is skipped, so the script can be stopped and
restarted.

    python pct_sweep.py --T 3 --pct 50 100 --out ../results/pct_sweep/pct_sweep_results.csv
    python pct_sweep.py --T 1 --pct 2 --lbfgs 20000 --seed 7
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from lorenz_pinn_full import Lorenz_PINN_solver

DT = 1e-4
EPOCHS = 10000
COLLOC_PER_T = 2000        # Adam physics points per unit of time
COLLOC_LBFGS_PER_T = 4000  # L-BFGS physics points per unit of time


def n_reference(T):
    return int(round(T / DT)) + 1


def n_points(pct, T):
    """pct % of the reference points, rounded half up (37.5 -> 38)."""
    return max(1, int(np.floor(pct / 100 * n_reference(T) + 0.5)))


SCIPY_MAXFUN = 15000  # SciPy's default cap on L-BFGS loss evaluations


def set_maxfun(maxfun):
    """Raise SciPy's loss-evaluation cap, which otherwise stops L-BFGS near 14,000 steps."""
    import scipy.optimize
    original = scipy.optimize.minimize

    def minimize(*args, options=None, **kw):
        options = dict(options or {}, maxfun=maxfun)
        return original(*args, options=options, **kw)
    scipy.optimize.minimize = minimize


def run_name(T, pct, lbfgs, seed, maxfun=SCIPY_MAXFUN):
    name = f"T{T:g}_pct{pct:g}"
    if lbfgs != 5000:
        name += f"_lbfgs{lbfgs}"
    if maxfun != SCIPY_MAXFUN:
        name += f"_maxfun{maxfun}"
    if seed != 99:
        name += f"_s{seed}"
    return name


class Pct_Solver(Lorenz_PINN_solver):
    """Lorenz_PINN_solver with Keras seeding pinned and an exact 100% option."""

    def __init__(self, **kw):
        super().__init__(**kw)
        tf.keras.utils.set_random_seed(self.seed)

    def sample_system(self, dt=DT, n_sample=200):
        current_df = self.lorenz_df.loc[self.lorenz_df["dt"] == dt, ].reset_index(drop=True)
        if n_sample >= len(current_df):
            # 100%: every reference point (equal-width bins would leave a few empty)
            self.sample_list.append(current_df.copy())
            return
        super().sample_system(dt, n_sample)

    # -------- checkpoints, so a long run can resume after the machine restarts --------
    # ckpt_path is set by run(). A checkpoint holds the flat weights and how many
    # L-BFGS steps were done. It is written after Adam and every ckpt_every L-BFGS
    # steps. An uninterrupted run is unchanged; a resumed run restarts L-BFGS from
    # the saved weights (its curvature memory starts fresh at that point).
    ckpt_path = None
    ckpt_every = 5000

    def _flat(self):
        return tf.concat([tf.reshape(v, [-1]) for v in self.train_vars], axis=0).numpy()

    def _set_flat(self, flat):
        sizes = [int(np.prod(v.shape)) for v in self.train_vars]
        for v, part in zip(self.train_vars, tf.split(tf.constant(flat, dtype=self.train_vars[0].dtype), sizes)):
            v.assign(tf.reshape(part, v.shape))

    def _rel_l2_now(self):
        ref = self.lorenz_df.loc[self.lorenz_df["dt"] == self.train_dt]
        pred = self.predict_state(tf.constant(ref[["t"]].values, dtype=self.state_ic.dtype)).numpy()
        true = ref[["x_pos", "y_pos", "z_pos"]].values
        return float(np.linalg.norm(pred - true) / np.linalg.norm(true))

    def _save_ckpt(self, done, flat=None):
        if self.ckpt_path is None:
            return
        flat = self._flat() if flat is None else flat
        tmp = self.ckpt_path + ".tmp.npz"
        np.savez(tmp, flat=flat, done=done)
        os.replace(tmp, self.ckpt_path)
        if self.ckpt_log is not None:
            with open(self.ckpt_log, "a") as f:
                f.write(f"{done},{self._rel_l2_now():.6e},{time.time():.0f}\n")

    ckpt_log = None

    def train(self, epochs=10000, print_every=1000):
        self.first_stage_epochs = epochs
        self.lbfgs_done = 0
        self.resumed_at = None
        if self.ckpt_path and os.path.exists(self.ckpt_path):
            ck = np.load(self.ckpt_path)
            self._set_flat(ck["flat"])
            self.lbfgs_done = self.resumed_at = int(ck["done"])
            print(f"resumed from checkpoint after Adam + {self.lbfgs_done} L-BFGS steps", flush=True)
        else:
            cap = self.lbfgs_iters
            self.lbfgs_iters = 0          # Adam only; L-BFGS runs below
            super().train(epochs, print_every)
            self.lbfgs_iters = cap
            self._save_ckpt(0)
        remaining = self.lbfgs_iters - self.lbfgs_done
        if remaining > 0:
            self.train_lbfgs_ckpt(remaining, start_epoch=epochs + self.lbfgs_done, print_every=print_every)
        self.loss_df = pd.DataFrame(self.loss_history)

    def train_lbfgs_ckpt(self, iters, start_epoch, print_every):
        """Same L-BFGS as Lorenz_PINN_solver.train_lbfgs, plus checkpoints."""
        from scipy.optimize import minimize
        self.w_phys.assign(self.w_phys_target)
        t_colloc = self.fixed_colloc()

        @tf.function
        def loss_and_grad():
            with tf.GradientTape() as tape:
                losses = self.compute_loss(t_colloc)
            grads = tape.gradient(losses[0], self.train_vars)
            return losses, tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

        last = {}

        def fun(flat):
            self._set_flat(flat)
            losses, grad = loss_and_grad()
            last["losses"] = [float(l) for l in losses]
            return last["losses"][0], grad.numpy()

        def callback(xk):
            self.lbfgs_done += 1
            total, l_data, l_phys, l_ic = last["losses"]
            self.loss_history.append({"epoch": self.first_stage_epochs + self.lbfgs_done,
                                      "total": total, "data": l_data, "physics": l_phys, "ic": l_ic})
            if self.lbfgs_done % print_every == 0:
                print(f"lbfgs {self.lbfgs_done:6d} | total {total:.3e} | data {l_data:.3e}"
                      f" | physics {l_phys:.3e} | ic {l_ic:.3e}", flush=True)
            if self.lbfgs_done % self.ckpt_every == 0:
                self._save_ckpt(self.lbfgs_done, flat=np.array(xk))
                self._set_flat(xk)

        result = minimize(fun, self._flat(), jac=True, method="L-BFGS-B", callback=callback,
                          options={"maxiter": iters, "maxcor": 50, "ftol": 1e-16, "gtol": 1e-12})
        self._set_flat(result.x)
        print(f"L-BFGS stopped after {self.lbfgs_done} steps in all: {result.message}", flush=True)


def run(T, pct, lbfgs=5000, seed=99, epochs=EPOCHS, print_every=2500, maxfun=SCIPY_MAXFUN,
        ckpt_dir=None):
    n = n_points(pct, T)
    start = time.time()
    s = Pct_Solver(T=T, dt=[DT], method="rk4", optimizer="adam", learning_rate=1e-2,
                   lbfgs_iters=lbfgs, seed=seed,
                   n_colloc=int(round(COLLOC_PER_T * T)),
                   n_colloc_lbfgs=int(round(COLLOC_LBFGS_PER_T * T)))
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        name = run_name(T, pct, lbfgs, seed, maxfun)
        s.ckpt_path = os.path.join(ckpt_dir, name + ".npz")
        s.ckpt_log = os.path.join(ckpt_dir, name + "_progress.csv")
    row = dict(name=run_name(T, pct, lbfgs, seed, maxfun), maxfun=maxfun, T=T, pct_reference=pct, n_data=n,
               n_reference=n_reference(T), rel_l2=np.nan, max_abs=np.nan, final_loss=np.nan,
               lbfgs_steps=0, resumed_at=None, lbfgs_cap=lbfgs, n_colloc=s.n_colloc,
               n_colloc_lbfgs=s.n_colloc_lbfgs, seed=seed)
    try:
        s.pinn_pipeline(dt=DT, n_sample=n, epochs=epochs, print_every=print_every)
        df = s.pred_df
        ref_3d = np.sqrt((df[["x_pos", "y_pos", "z_pos"]] ** 2).sum(axis=1))
        row.update(rel_l2=float(np.linalg.norm(df["err_3d"]) / np.linalg.norm(ref_3d)),
                   max_abs=float(df[["x_err", "y_err", "z_err"]].abs().values.max()),
                   final_loss=s.loss_history[-1]["total"],
                   lbfgs_steps=s.lbfgs_done, resumed_at=s.resumed_at)
    except Exception as e:  # keep going; the row records that the run produced nothing
        print(f"run failed: {e!r}")
    row["seconds"] = round(time.time() - start)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=float, nargs="+", required=True)
    p.add_argument("--pct", type=float, nargs="+", required=True)
    p.add_argument("--lbfgs", type=int, default=5000)
    p.add_argument("--seed", type=int, nargs="+", default=[99])
    p.add_argument("--maxfun", type=int, default=SCIPY_MAXFUN,
                   help="L-BFGS loss-evaluation cap (SciPy default 15000)")
    p.add_argument("--out", default="pct_sweep_results.csv")
    p.add_argument("--ckpt-dir", default=None, help="save/resume checkpoints here")
    a = p.parse_args()
    if a.maxfun != SCIPY_MAXFUN:
        set_maxfun(a.maxfun)

    for T in a.T:
        for pct in a.pct:
            for seed in a.seed:
                name = run_name(T, pct, a.lbfgs, seed, a.maxfun)
                if os.path.exists(a.out) and name in set(pd.read_csv(a.out)["name"]):
                    print(f"skip {name} (already in {a.out})")
                    continue
                print(f"=== {name}: {n_points(pct, T)} of {n_reference(T)} points ===", flush=True)
                row = run(T, pct, a.lbfgs, seed, maxfun=a.maxfun, ckpt_dir=a.ckpt_dir)
                print(f"=== {name}: rel L2 {row['rel_l2']:.3e} in {row['seconds']} s ===", flush=True)
                out = pd.DataFrame([row])
                out.to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)


if __name__ == "__main__":
    main()
