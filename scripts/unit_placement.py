"""Which part of each 1-unit window needs data?

For window k = [k, k+1] of the true path, train one window on its own, starting
from the TRUE state at t = k (so windows don't affect each other), with a single
0.05-long data piece centered at local position p (or no data at all), physics
on the whole window. Score the window's relative L2 error against RK4.
"""
import json, os, sys, time
import numpy as np
import pandas as pd

from time_marching import Lorenz_Window
from lorenz_pinn_full import Lorenz
from interval_validation import TUNED, TUNED_EPOCHS

k, pos, lbfgs, out = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
dt = TUNED["dt"][0]
ref = Lorenz(T=k + 1.0, dt=[dt], method="rk4")
ref.lorenz_pipeline()
df = ref.lorenz_df.reset_index(drop=True)
w = df[df.t >= k - 1e-9].copy()
w["t"] = (w["t"] - k).round(10)
w = w.reset_index(drop=True)
ic = w.iloc[0][["x_pos", "y_pos", "z_pos"]].values.astype(float)
local = [] if pos == "none" else [(float(pos) - 0.025, float(pos) + 0.025)]

t0 = time.time()
s = Lorenz_Window(w, local, T=1.0, x0=ic[0], y0=ic[1], z0=ic[2], dt=[dt], method="rk4",
                  optimizer=TUNED["optimizer"], learning_rate=TUNED["learning_rate"],
                  lbfgs_iters=lbfgs, seed=99, n_colloc=2000, n_colloc_lbfgs=4000)
s.pinn_pipeline(dt=dt, n_sample=200, epochs=TUNED_EPOCHS, print_every=5000)
p = s.pred_df
ref3 = np.sqrt((p[["x_pos", "y_pos", "z_pos"]] ** 2).sum(axis=1))
row = {"unit": k + 1, "start": k, "piece_center": pos, "lbfgs_cap": lbfgs,
       "rel_l2": float(np.linalg.norm(p.err_3d) / np.linalg.norm(ref3)),
       "max_abs": float(p[["x_err", "y_err", "z_err"]].abs().values.max()),
       "end_err3d": float(p.err_3d.iloc[-1]), "seconds": round(time.time() - t0)}
print("ROW", json.dumps(row))
d = pd.DataFrame([row])
if os.path.exists(out):
    d = pd.concat([pd.read_csv(out), d], ignore_index=True)
d.to_csv(out, index=False)
