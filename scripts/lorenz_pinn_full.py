import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Use 64-bit floats everywhere (about 16 significant digits instead of 7),
# so round-off stays far below the 1e-4 accuracy target.
DTYPE = tf.float64
tf.keras.backend.set_floatx("float64")  # Keras layers and weights follow this
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.animation import FuncAnimation


# make linear space function (unchanged)

def linear_space(start, end, steps):

    step_length = (end - start) / steps

    even_spaced_nums = [start]

    # keeps track if we have reached
    tracker = start

    # add values to the even_spaced_nums list
    while tracker < end:
        tracker = tracker + step_length
        even_spaced_nums.append(tracker)

    return np.array(even_spaced_nums)  # return whole list as numpy array


class Lorenz:

    # CHANGED: T and the initial condition now live on the object, so the
    # PINN can read them later (it needs T for scaling and x0, y0, z0 for the IC loss).
    def __init__(self, sigma=10., beta=8/3, rho=28.,
                 dt=[.001, .0001, .00001, .000001],
                 T=1.0, x0=1., y0=1., z0=1., method="rk4"):
        """
        method : "rk4" (default) or "euler". RK4's error shrinks like dt^4,
                 so at dt = 1e-4 its error is far below 1e-8. Euler's
                 error shrinks like dt and is about 3e-3 at dt = 1e-4,
                 which is far too rough to train or grade against for 1e-4.
        """
        if method not in ("rk4", "euler"):
            raise ValueError("method must be 'rk4' or 'euler'")

        self.method = method
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.dt_list = dt
        self.T = T
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.lorenz_list = []
        self.sample_list = []
        self.lorenz_df = pd.DataFrame(data=None)

    def __str__(self):

        output_string = "Sigma: " + str(self.sigma) + "\nRho: " + str(self.rho) + "\nBeta: " + str(self.beta)
        return output_string

    def derivatives(self, x, y, z):

        dx = self.sigma * (y - x)
        dy = self.rho * x - y - x * z
        dz = x * y - self.beta * z
        return dx, dy, dz

    def rk4_step(self, x, y, z, dt):

        # Four slope estimates across the step, combined in a weighted average
        k1 = self.derivatives(x, y, z)
        k2 = self.derivatives(x + dt / 2 * k1[0], y + dt / 2 * k1[1], z + dt / 2 * k1[2])
        k3 = self.derivatives(x + dt / 2 * k2[0], y + dt / 2 * k2[1], z + dt / 2 * k2[2])
        k4 = self.derivatives(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2])

        x += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        y += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        z += dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        return x, y, z

    def get_ic_list(self, dt=1/10000):

        # CHANGED: round() instead of int(). 1 / 1e-5 is 99999.99999999999
        # in floating point, and int() would silently drop the last step.
        N = int(round(self.T / dt))
        x, y, z = self.x0, self.y0, self.z0

        self.x_list = [x]
        self.y_list = [y]
        self.z_list = [z]
        self.t_list = [0.0]  # NEW: the PINN takes time as its input

        for i in range(N):

            if self.method == "rk4":
                x, y, z = self.rk4_step(x, y, z, dt)
            else:
                dx, dy, dz = self.derivatives(x, y, z)
                x += dt * dx
                y += dt * dy
                z += dt * dz

            self.x_list.append(x)
            self.y_list.append(y)
            self.z_list.append(z)
            self.t_list.append((i + 1) * dt)  # step count * dt avoids drift from repeated adding

    def get_lorenz_system(self):

        for dt in self.dt_list:

            self.get_ic_list(dt = dt)
            n = len(self.x_list)

            self.lorenz_list.append({"sigma": [self.sigma] * n,
                                     "beta": [self.beta] * n,
                                     "rho": [self.rho] * n,
                                     "t": self.t_list,  # NEW column
                                     "x_pos": self.x_list,
                                     "y_pos": self.y_list,
                                     "z_pos": self.z_list,
                                     "dt": [dt] * n})

    def get_dataframe(self):

        for current_list in self.lorenz_list:

            self.lorenz_df = pd.concat([self.lorenz_df, pd.DataFrame(current_list)])

    def lorenz_pipeline(self):

        self.get_lorenz_system()
        self.get_dataframe()


class Lorenz_PINN_solver(Lorenz):

    def __init__(self, hidden_layers=4, units=64, learning_rate=1e-2,
                 n_colloc=2000, loss_weights=(1.0, 1.0, 10.0),
                 warmup_epochs=1000, inverse=False, seed=99,
                 optimizer="adam", lr_final_fraction=0.01, lbfgs_iters=20000,
                 n_colloc_lbfgs=4000, hard_ic=True, **lorenz_kwargs):
        """
        hidden_layers, units : size of the network
        n_colloc             : physics (collocation) points drawn each epoch
        loss_weights         : (data, physics, initial condition)
        warmup_epochs        : epochs of data-only training, after which the
                               physics weight ramps up over another warmup_epochs
        inverse              : if True, sigma / rho / beta become trainable
        optimizer            : "adam" or "sgd" (plain gradient descent)
        lr_final_fraction    : the learning rate decays smoothly (cosine) from
                               learning_rate to learning_rate * this by the last
                               epoch. None keeps it constant.
        lbfgs_iters          : L-BFGS steps after the first-stage optimizer
                               finishes, to polish the last digits. 0 turns it off.
        n_colloc_lbfgs       : fixed physics points used during L-BFGS
        hard_ic              : if True, the network output is built so it equals
                               (x0, y0, z0) at t = 0 exactly
        lorenz_kwargs        : anything Lorenz takes (sigma, T, dt, x0, method, ...)
        """
        super().__init__(**lorenz_kwargs)

        tf.random.set_seed(seed)
        self.seed = seed
        self.hidden_layers = hidden_layers
        self.units = units
        self.learning_rate = learning_rate
        self.n_colloc = n_colloc
        self.w_data, self.w_phys_target, self.w_ic = loss_weights
        self.warmup_epochs = warmup_epochs
        if optimizer not in ("adam", "sgd"):
            raise ValueError("optimizer must be 'adam' or 'sgd'")
        self.optimizer_name = optimizer
        self.lr_final_fraction = lr_final_fraction
        self.lbfgs_iters = lbfgs_iters
        self.n_colloc_lbfgs = n_colloc_lbfgs
        self.hard_ic = hard_ic

        # The physics weight changes during training, so it is a tf.Variable.
        # A plain Python float would get frozen into the tf.function graph.
        self.w_phys = tf.Variable(0.0, dtype=DTYPE, trainable=False)
        self.inverse = inverse
        self.loss_history = []

    # ------------------------------------------------------------------
    # 1. Sampling
    # ------------------------------------------------------------------

    def sample_system(self, dt=.0001, n_sample = 200): # 

        current_df = self.lorenz_df.loc[self.lorenz_df["dt"] == dt, ].reset_index(drop=True)

        # CHANGED: stratified sampling with a fixed count instead of frac.
        # Cut [0, T] into n_sample equal time bins and take one point from each,
        # so there are no big gaps where the PINN is left guessing.
        
        time_bins = pd.cut(current_df["t"], bins = n_sample) # weighted sample for more [.5 - 1]
        sample = (current_df.groupby(time_bins, observed = True)
                            .sample(n=1, random_state = self.seed)
                            .sort_values("t")
                            .reset_index(drop=True))

        self.sample_list.append(sample)

    def get_samples(self, n_sample=200):

        for current_dt in self.dt_list:

            self.sample_system(current_dt, n_sample)

    def sample_pipeline(self, n_sample=200):

        self.lorenz_pipeline()
        self.get_samples(n_sample)

    # ------------------------------------------------------------------
    # 2. Turn one dt's sample into tensors
    # ------------------------------------------------------------------

    def prepare_training_data(self, dt=.0001):

        # sample_list is in the same order as dt_list
        sample = self.sample_list[self.dt_list.index(dt)]
        self.train_dt = dt

        cols = ["x_pos", "y_pos", "z_pos"]

        # Data points: shape (n, 1) for time, (n, 3) for the state
        self.t_data = tf.constant(sample[["t"]].values, dtype=DTYPE)
        self.state_data = tf.constant(sample[cols].values, dtype=DTYPE)

        # Initial condition: a single point at t = 0
        self.t_ic = tf.zeros((1, 1), dtype=DTYPE)
        self.state_ic = tf.constant([[self.x0, self.y0, self.z0]], dtype=DTYPE)

        # Output scaling. The network predicts roughly unit-sized numbers and we
        # map them back with mean/std. Without this, z (up to ~45) would dominate.
        self.state_mean = tf.constant(sample[cols].mean().values.reshape(1, 3), dtype=DTYPE)
        self.state_std = tf.constant(sample[cols].std().values.reshape(1, 3), dtype=DTYPE)

    # ------------------------------------------------------------------
    # 3. Network and physics parameters
    # ------------------------------------------------------------------

    def build_network(self, initial_params=None):
        """
        initial_params : optional (sigma, rho, beta) starting guesses,
                         only meaningful when inverse=True.
        """
        inputs = tf.keras.Input(shape=(1,))
        h = inputs
        for _ in range(self.hidden_layers):
            # tanh, not ReLU: the physics loss needs smooth derivatives in t
            h = tf.keras.layers.Dense(self.units, activation = "tanh",
                                      kernel_initializer = "glorot_normal")(h)
        outputs = tf.keras.layers.Dense(3)(h)  # x, y, z (scaled)
        self.model = tf.keras.Model(inputs, outputs)

        # sigma, rho, beta as tf.Variables. With inverse=False they are frozen
        # constants; with inverse=True the optimizer learns them from the data.
        if initial_params is None:
            initial_params = (self.sigma, self.rho, self.beta)
        s0, r0, b0 = initial_params
        self.sigma_tf = tf.Variable(s0, dtype=DTYPE, trainable=self.inverse)
        self.rho_tf = tf.Variable(r0, dtype=DTYPE, trainable=self.inverse)
        self.beta_tf = tf.Variable(b0, dtype=DTYPE, trainable=self.inverse)

        self.train_vars = list(self.model.trainable_variables)
        if self.inverse:
            self.train_vars += [self.sigma_tf, self.rho_tf, self.beta_tf]


    def predict_state(self, t):
        """Physical time in, physical (x, y, z) out.

        Scaling happens inside this function, so when GradientTape
        differentiates it with respect to t, the chain rule through the
        scaling is handled automatically.
        """
        t_norm = 2.0 * t / self.T - 1.0  # [0, T] -> [-1, 1]
        out = self.model(t_norm)

        if self.hard_ic:
            # (t / T) is 0 at t = 0, so the output is exactly the initial
            # condition there no matter what the network's weights are.
            return self.state_ic + (t / self.T) * out * self.state_std
        return out * self.state_std + self.state_mean

    # ------------------------------------------------------------------
    # 4. Physics residual
    # ------------------------------------------------------------------

    def physics_residual(self, t):

        # Inner tape: derivatives of the state with respect to time.
        # persistent=True lets us call .gradient three times.
        
        # the PINN does not know the initial cond
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            state = self.predict_state(t)
            x = state[:, 0:1]
            y = state[:, 1:2]
            z = state[:, 2:3]

        dx_dt = tape.gradient(x, t)
        dy_dt = tape.gradient(y, t)
        dz_dt = tape.gradient(z, t)
        del tape

        # Right-hand side of the Lorenz equations
        f_x = self.sigma_tf * (y - x)
        f_y = self.rho_tf * x - y - x * z
        f_z = x * y - self.beta_tf * z

        residual = tf.concat([dx_dt - f_x, dy_dt - f_y, dz_dt - f_z], axis=1)

        # Put the residual in the same scaled units the network works in.
        # Raw derivatives can be in the hundreds, which would swamp the data loss.
        return residual * (self.T / 2.0) / self.state_std

    # ------------------------------------------------------------------
    # 5. Loss and training
    # ------------------------------------------------------------------

    def compute_loss(self, t_colloc=None):

        # Data loss: match the sampled reference points
        pred_data = self.predict_state(self.t_data)
        loss_data = tf.reduce_mean(tf.square((pred_data - self.state_data) / self.state_std))

        # Physics loss: fresh random collocation times every step
        # (L-BFGS passes in a fixed set, since it needs the same loss each call)
        if t_colloc is None:
            t_colloc = tf.random.uniform((self.n_colloc, 1), 0.0, self.T, dtype=DTYPE)
        loss_phys = tf.reduce_mean(tf.square(self.physics_residual(t_colloc)))

        # Initial condition loss
        pred_ic = self.predict_state(self.t_ic)
        loss_ic = tf.reduce_mean(tf.square((pred_ic - self.state_ic) / self.state_std))

        total = self.w_data * loss_data + self.w_phys * loss_phys + self.w_ic * loss_ic # Plot each one of these
        
        return total, loss_data, loss_phys, loss_ic

    def _train_step(self):

        # Outer tape: derivatives of the loss with respect to the weights
        with tf.GradientTape() as tape:
            total, loss_data, loss_phys, loss_ic = self.compute_loss()

        grads = tape.gradient(total, self.train_vars)
        self.optimizer.apply_gradients(zip(grads, self.train_vars))
        return total, loss_data, loss_phys, loss_ic

    def build_optimizer(self, epochs):

        lr = self.learning_rate
        if self.lr_final_fraction is not None:
            # Big steps early to find the right region, small steps late to settle in
            lr = tf.keras.optimizers.schedules.CosineDecay(
                self.learning_rate, decay_steps=epochs, alpha=self.lr_final_fraction)

        if self.optimizer_name == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Wrap the train step in tf.function after the optimizer exists,
        # so the compiled graph uses this run's optimizer and tensors.
        self.train_step = tf.function(self._train_step)

    def train(self, epochs=10000, print_every=1000):

        self.first_stage_epochs = epochs  # L-BFGS steps are numbered after these
        self.build_optimizer(epochs)

        for epoch in range(1, epochs + 1):

            # Physics warm-up: fit the data first, then phase in the physics.
            # Turning the physics loss on at full strength from the start tends
            # to trap the network in a smooth, wrong solution that satisfies
            # the ODE loosely and ignores the data.
            if self.warmup_epochs > 0:
                ramp = (epoch - self.warmup_epochs) / self.warmup_epochs
                ramp = min(1.0, max(0.0, ramp))  # 0 for the first phase, then 0 -> 1
            else:
                ramp = 1.0
            self.w_phys.assign(self.w_phys_target * ramp)

            total, l_data, l_phys, l_ic = self.train_step()

            record = {"epoch": epoch, "total": float(total), "data": float(l_data),
                      "physics": float(l_phys), "ic": float(l_ic)}
            if self.inverse:
                record.update({"sigma": float(self.sigma_tf), "rho": float(self.rho_tf),
                               "beta": float(self.beta_tf)})
            self.loss_history.append(record)

            if epoch % print_every == 0 or epoch == 1:
                msg = (f"epoch {epoch:6d} | total {record['total']:.3e} | data {record['data']:.3e}"
                       f" | physics {record['physics']:.3e} | ic {record['ic']:.3e}")
                if self.inverse:
                    msg += f" | sigma {record['sigma']:.3f} rho {record['rho']:.3f} beta {record['beta']:.3f}"
                print(msg)

        if self.lbfgs_iters > 0:
            self.train_lbfgs(start_epoch=epochs, print_every=print_every)

        self.loss_df = pd.DataFrame(self.loss_history)

    def train_lbfgs(self, start_epoch=0, print_every=1000):
        """Second training stage with L-BFGS (via scipy).

        L-BFGS uses the curvature of the loss, not just its slope, so near a
        minimum it takes much better steps than Adam or plain gradient descent.
        It needs the exact same loss on every call, so the physics points are
        fixed (an evenly spaced grid) instead of re-drawn each step.
        """
        from scipy.optimize import minimize

        self.w_phys.assign(self.w_phys_target)
        t_colloc = tf.reshape(tf.linspace(tf.constant(0.0, DTYPE), self.T, self.n_colloc_lbfgs), (-1, 1))
        shapes = [v.shape for v in self.train_vars]
        sizes = [int(np.prod(sh)) for sh in shapes]

        def set_weights(flat):
            parts = tf.split(tf.constant(flat, dtype=DTYPE), sizes)
            for v, part, sh in zip(self.train_vars, parts, shapes):
                v.assign(tf.reshape(part, sh))

        @tf.function
        def loss_and_grad():
            with tf.GradientTape() as tape:
                losses = self.compute_loss(t_colloc)
            grads = tape.gradient(losses[0], self.train_vars)
            flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
            return losses, flat

        last = {}

        def fun(flat):
            set_weights(flat)
            losses, grad = loss_and_grad()
            last["losses"] = [float(l) for l in losses]
            return last["losses"][0], grad.numpy()

        step = [start_epoch]

        def callback(_):
            step[0] += 1
            total, l_data, l_phys, l_ic = last["losses"]
            record = {"epoch": step[0], "total": total, "data": l_data,
                      "physics": l_phys, "ic": l_ic}
            if self.inverse:
                record.update({"sigma": float(self.sigma_tf), "rho": float(self.rho_tf),
                               "beta": float(self.beta_tf)})
            self.loss_history.append(record)
            if (step[0] - start_epoch) % print_every == 0:
                print(f"lbfgs {step[0] - start_epoch:6d} | total {total:.3e} | data {l_data:.3e}"
                      f" | physics {l_phys:.3e} | ic {l_ic:.3e}")

        x0 = tf.concat([tf.reshape(v, [-1]) for v in self.train_vars], axis=0).numpy()
        result = minimize(fun, x0, jac=True, method="L-BFGS-B", callback=callback,
                          options={"maxiter": self.lbfgs_iters, "maxcor": 50,
                                   "ftol": 1e-16, "gtol": 1e-12})
        set_weights(result.x)
        print(f"L-BFGS stopped after {result.nit} steps: {result.message}")

    # ------------------------------------------------------------------
    # 6. Predict and compare with the full reference trajectory
    # ------------------------------------------------------------------

    def predict(self, every=1):
        """Evaluate the PINN at the times of the full reference run it was trained on.

        every : keep every k-th row (useful for dt = 1e-6, which has a million rows).
        """
        ref = (self.lorenz_df.loc[self.lorenz_df["dt"] == self.train_dt]
                             .iloc[::every]
                             .reset_index(drop=True))

        t = tf.constant(ref[["t"]].values, dtype=DTYPE)
        pred = self.predict_state(t).numpy()

        self.pred_df = ref[["t", "x_pos", "y_pos", "z_pos"]].copy()
        self.pred_df["x_pinn"] = pred[:, 0]
        self.pred_df["y_pinn"] = pred[:, 1]
        self.pred_df["z_pinn"] = pred[:, 2]

        # Relative L2 error per variable: ||pinn - reference|| / ||reference||
        for v in ["x", "y", "z"]:
            true = self.pred_df[f"{v}_pos"].values
            err = np.linalg.norm(self.pred_df[f"{v}_pinn"].values - true) / np.linalg.norm(true)
            print(f"relative L2 error {v}: {err:.3e}")

        max_abs = np.abs(self.pred_df[["x_pinn", "y_pinn", "z_pinn"]].values
                         - self.pred_df[["x_pos", "y_pos", "z_pos"]].values).max()
        print(f"max absolute error (worst single point): {max_abs:.3e}")

        return self.pred_df

    # ------------------------------------------------------------------
    # 7. Everything in order
    # ------------------------------------------------------------------

    def pinn_pipeline(self, dt=.0001, n_sample=200, epochs=10000,
                      print_every=1000, initial_params=None):

        self.sample_pipeline(n_sample)
        self.prepare_training_data(dt)
        self.build_network(initial_params)
        self.train(epochs, print_every)
        return self.predict()


class Lorenz_Visuals:
    """Plots for a trained Lorenz_PINN_solver.

    Takes the solver *after* training and predict() have run, and reads from:
        solver.pred_df  : t, x_pos/y_pos/z_pos (reference), x_pinn/y_pinn/z_pinn (PINN)
        solver.loss_df  : one row per epoch with total, data, physics, ic losses
        solver.sample_list / dt_list / train_dt : the points the PINN trained on

    Every plot method returns its figure, and saves it if you pass save_path.
    """

    VARS = ["x", "y", "z"]
    COLORS = {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}
    LOSS_COLORS = {"total": "black", "data": "tab:blue", "physics": "tab:red", "ic": "tab:purple"}

    def __init__(self, solver, save_dir=None):

        if not hasattr(solver, "pred_df"):
            raise ValueError("Run solver.predict() (or pinn_pipeline) before making visuals.")

        self.solver = solver
        self.pred_df = solver.pred_df
        self.loss_df = solver.loss_df
        self.train_dt = solver.train_dt

        # The 200 points the PINN actually trained on, so plots can mark them
        self.sample_df = solver.sample_list[solver.dt_list.index(solver.train_dt)]

        # Physics weight is 0 until warmup_epochs, ramps until 2 * warmup_epochs
        self.warmup_epochs = getattr(solver, "warmup_epochs", 0)

        # Rows after this epoch number are L-BFGS steps, not Adam/SGD epochs
        self.first_stage_epochs = getattr(solver, "first_stage_epochs", None)

        self.save_dir = save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _slice(self, n_steps):
        """First n_steps rows of pred_df (all rows if None).

        A "step" is one row of pred_df. With dt = 1e-4 and predict(every=1),
        n_steps=2500 means the first 0.25 seconds.
        """
        if n_steps is None:
            return self.pred_df
        return self.pred_df.iloc[:n_steps]

    def _save(self, fig, save_path):
        if save_path is None:
            return
        if self.save_dir is not None and not os.path.isabs(save_path):
            save_path = os.path.join(self.save_dir, save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    def _log_labels(self, ax):
        """Label minor log ticks too. When the data spans only one or two
        decades, matplotlib otherwise shows almost no numbers on the y-axis."""
        ax.yaxis.set_minor_formatter(
            ticker.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(2, 0.5)))
        ax.tick_params(axis="y", which="minor", labelsize=7)

    def _shade_warmup(self, ax):
        """Shade the epochs where the physics weight was 0 or still ramping."""
        w = self.warmup_epochs
        if w and w > 0:
            ax.axvspan(1, w, color="gray", alpha=0.12, label="data only (physics off)")
            ax.axvspan(w, 2 * w, color="gray", alpha=0.06, label="physics ramping up")

    def _shade_lbfgs(self, ax):
        """Shade the L-BFGS stage, which comes after the Adam/SGD epochs."""
        start = self.first_stage_epochs
        end = self.loss_df["epoch"].max()
        if start is not None and end > start:
            ax.axvspan(start, end, color="tab:green", alpha=0.08, label="L-BFGS")

    # ------------------------------------------------------------------
    # 1 & 2. 3D trajectories: reference and predicted
    # ------------------------------------------------------------------

    def plot_reference_3d(self, n_steps=None, show_samples=True, save_path=None):
        """Reference trajectory in (x, y, z) space."""

        df = self._slice(n_steps)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(projection="3d")

        ax.plot(df["x_pos"], df["y_pos"], df["z_pos"], color="black", lw=1.2, label="reference")

        if show_samples:
            s = self.sample_df[self.sample_df["t"] <= df["t"].iloc[-1]]
            ax.scatter(s["x_pos"], s["y_pos"], s["z_pos"], color="tab:red", s=8,
                       label=f"training samples (n={len(s)})")

        ax.scatter(*df[["x_pos", "y_pos", "z_pos"]].iloc[0], color="green", s=40, label="start")
        self._label_3d(ax, f"Reference solution ({self.solver.method.upper()}, dt = {self.train_dt:g})")
        self._save(fig, save_path)
        return fig

    def plot_predicted_3d(self, n_steps=None, show_reference=False, save_path=None):
        """PINN predicted trajectory in (x, y, z) space.

        show_reference=True draws the reference path faintly underneath for comparison.
        """
        df = self._slice(n_steps)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(projection="3d")

        if show_reference:
            ax.plot(df["x_pos"], df["y_pos"], df["z_pos"], color="gray", lw=1, alpha=0.5, label="reference")

        ax.plot(df["x_pinn"], df["y_pinn"], df["z_pinn"], color="tab:red", lw=1.4, label="PINN prediction")
        ax.scatter(*df[["x_pinn", "y_pinn", "z_pinn"]].iloc[0], color="green", s=40, label="start")
        self._label_3d(ax, "Predicted solution (PINN)")
        self._save(fig, save_path)
        return fig

    def _label_3d(self, ax, title):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)

    # ------------------------------------------------------------------
    # 3. x, y, z over time for a chosen number of time steps
    # ------------------------------------------------------------------

    def plot_time_series(self, n_steps=None, show_samples=True, save_path=None):
        """x(t), y(t), z(t): PINN vs reference, one panel per variable.

        n_steps limits the plot to the first n_steps rows, e.g. to zoom in
        on the fast transient around t = 0.3.
        """
        df = self._slice(n_steps)
        t_end = df["t"].iloc[-1]
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        for ax, v in zip(axes, self.VARS):
            ax.plot(df["t"], df[f"{v}_pos"], color="black", lw=1.5, label="reference")
            ax.plot(df["t"], df[f"{v}_pinn"], color=self.COLORS[v], lw=1.5, ls="--", label="PINN")
            if show_samples:
                s = self.sample_df[self.sample_df["t"] <= t_end]
                ax.scatter(s["t"], s[f"{v}_pos"], color="tab:red", s=8, zorder=3, label="training samples")
            ax.set_ylabel(v)
            ax.grid(alpha=0.3)

        axes[0].legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("t")
        axes[0].set_title(f"PINN vs reference over {len(df)} time steps (t = 0 to {t_end:.3f})")
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    def animate_prediction(self, n_steps=None, frames=200, interval=30, save_path=None):
        """Animate the PINN tracing its trajectory in 3D next to the reference.

        frames   : number of animation frames (n_steps rows are split evenly across them)
        save_path: ".gif" (needs pillow) or ".mp4" (needs ffmpeg). In a Jupyter
                   notebook you can display it with
                   from IPython.display import HTML; HTML(anim.to_jshtml())
        """
        df = self._slice(n_steps)
        idx = np.linspace(1, len(df), frames).astype(int)

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(projection="3d")

        # Fix the axes up front so the view doesn't jump as the line grows
        for setter, v in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], self.VARS):
            lo = min(df[f"{v}_pos"].min(), df[f"{v}_pinn"].min())
            hi = max(df[f"{v}_pos"].max(), df[f"{v}_pinn"].max())
            pad = 0.05 * (hi - lo)
            setter(lo - pad, hi + pad)

        ref_line, = ax.plot([], [], [], color="gray", lw=1, alpha=0.6, label="reference")
        pinn_line, = ax.plot([], [], [], color="tab:red", lw=1.5, label="PINN")
        head, = ax.plot([], [], [], "o", color="tab:red", ms=5)
        self._label_3d(ax, "")

        def update(k):
            n = idx[k]
            part = df.iloc[:n]
            ref_line.set_data_3d(part["x_pos"], part["y_pos"], part["z_pos"])
            pinn_line.set_data_3d(part["x_pinn"], part["y_pinn"], part["z_pinn"])
            last = part.iloc[-1]
            head.set_data_3d([last["x_pinn"]], [last["y_pinn"]], [last["z_pinn"]])
            ax.set_title(f"PINN prediction, t = {last['t']:.3f}")
            return ref_line, pinn_line, head

        anim = FuncAnimation(fig, update, frames=len(idx), interval=interval, blit=False)

        if save_path is not None:
            if self.save_dir is not None and not os.path.isabs(save_path):
                save_path = os.path.join(self.save_dir, save_path)
            anim.save(save_path, writer="pillow" if save_path.endswith(".gif") else "ffmpeg")
        return anim

    # ------------------------------------------------------------------
    # 4. Error between reference and prediction, log scale
    # ------------------------------------------------------------------

    def plot_errors(self, n_steps=None, show_samples=True, save_path=None):
        """|PINN - reference| for x, y, z over time on a log y-axis.

        Log scale because the error can span several orders of magnitude:
        tiny near the initial condition, much larger in the fast transient.
        Vertical ticks at the bottom mark where training samples were taken,
        so you can see whether error is lower right at the data points.
        """
        df = self._slice(n_steps)
        fig, ax = plt.subplots(figsize=(10, 4.5))

        for v in self.VARS:
            err = np.abs(df[f"{v}_pinn"] - df[f"{v}_pos"])
            # clip at a tiny floor so exact zeros don't break the log axis
            ax.semilogy(df["t"], np.maximum(err, 1e-12), color=self.COLORS[v], lw=1.2, label=f"|{v} error|")

        if show_samples:
            s = self.sample_df[self.sample_df["t"] <= df["t"].iloc[-1]]
            ax.plot(s["t"], np.zeros(len(s)), "|", color="tab:red", ms=8,
                    transform=ax.get_xaxis_transform(), label="training samples")

        ax.set_xlabel("t")
        ax.set_ylabel("absolute error (log scale)")
        ax.set_title("Pointwise error: PINN vs reference")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # 5. Total loss and each term, every epoch
    # ------------------------------------------------------------------

    def plot_losses(self, weighted=False, save_path=None):
        """Total loss and each term across all epochs (log scale).

        weighted=False shows the raw terms (what each loss actually is).
        weighted=True multiplies each by its weight, so you see how much
        each one contributes to the total the optimizer is minimizing.
        """
        df = self.loss_df
        s = self.solver
        fig, ax = plt.subplots(figsize=(10, 5))
        self._shade_warmup(ax)
        self._shade_lbfgs(ax)

        if weighted:
            # physics weight follows the warm-up ramp, so rebuild it per epoch
            w = self.warmup_epochs
            if w and w > 0:
                ramp = np.clip((df["epoch"] - w) / w, 0.0, 1.0)
            else:
                ramp = 1.0
            terms = {"data": s.w_data * df["data"],
                     "physics": s.w_phys_target * ramp * df["physics"],
                     "ic": s.w_ic * df["ic"]}
        else:
            terms = {"data": df["data"], "physics": df["physics"], "ic": df["ic"]}

        ax.semilogy(df["epoch"], df["total"], color=self.LOSS_COLORS["total"], lw=1.5, label="total")
        for name, values in terms.items():
            ax.semilogy(df["epoch"], values, color=self.LOSS_COLORS[name], lw=1, alpha=0.85, label=name)

        ax.set_xlabel("epoch (Adam/SGD), then L-BFGS step")
        ax.set_ylabel("loss (log scale)")
        ax.set_title("Loss terms per epoch" + (" (weighted)" if weighted else " (unweighted)"))
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # 6. Losses only at the epochs where the total hit a new minimum
    # ------------------------------------------------------------------

    def get_minima(self, start_epoch=None):
        """Rows of loss_df where the total loss reached a new lowest value.

        The raw loss bounces around from step to step (Adam steps and fresh
        random collocation points each epoch). Keeping only the epochs where
        the total beat every earlier epoch gives the clean "best so far" path.

        start_epoch defaults to the end of the warm-up. Before that the total
        loss isn't comparable, because the physics weight is still changing.
        """
        df = self.loss_df
        if start_epoch is None:
            start_epoch = 2 * self.warmup_epochs if self.warmup_epochs else 1
        df = df[df["epoch"] >= start_epoch]

        best_before = df["total"].cummin().shift(1, fill_value=np.inf)
        return df[df["total"] < best_before]

    def plot_loss_minima(self, start_epoch=None, save_path=None):
        """Total, physics, and data loss plotted only at new-minimum epochs."""
        minima = self.get_minima(start_epoch)
        fig, ax = plt.subplots(figsize=(10, 5))
        self._shade_lbfgs(ax)

        for name in ["total", "physics", "data"]:
            ax.semilogy(minima["epoch"], minima[name], marker="o", ms=3, lw=1,
                        color=self.LOSS_COLORS[name], label=name)

        ax.set_xlabel("epoch (Adam/SGD), then L-BFGS step")
        ax.set_ylabel("loss (log scale)")
        self._log_labels(ax)
        ax.set_title(f"Losses at new minima of the total loss ({len(minima)} of {len(self.loss_df)} epochs)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # Extra: parameter convergence for the inverse problem
    # ------------------------------------------------------------------

    def plot_parameters(self, true_params=(10.0, 28.0, 8 / 3), save_path=None):
        """sigma, rho, beta per epoch. Only meaningful when inverse=True."""
        if "sigma" not in self.loss_df.columns:
            raise ValueError("No parameter history: this solver was trained with inverse=False.")

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, name, true in zip(axes, ["sigma", "rho", "beta"], true_params):
            ax.plot(self.loss_df["epoch"], self.loss_df[name], color="tab:blue", label="learned")
            ax.axhline(true, color="black", ls="--", lw=1, label=f"true = {true:.3f}")
            ax.set_title(name)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    # ------------------------------------------------------------------
    # Everything at once
    # ------------------------------------------------------------------

    def plot_all(self, n_steps=None, show=True):
        """Make every static plot. Saves them too if save_dir was given."""
        save = self.save_dir is not None
        figs = {
            "reference_3d": self.plot_reference_3d(n_steps, save_path="reference_3d.png" if save else None),
            "predicted_3d": self.plot_predicted_3d(n_steps, show_reference=True,
                                                   save_path="predicted_3d.png" if save else None),
            "time_series": self.plot_time_series(n_steps, save_path="time_series.png" if save else None),
            "errors": self.plot_errors(n_steps, save_path="errors.png" if save else None),
            "losses": self.plot_losses(save_path="losses.png" if save else None),
            "loss_minima": self.plot_loss_minima(save_path="loss_minima.png" if save else None),
        }
        if "sigma" in self.loss_df.columns:
            figs["parameters"] = self.plot_parameters(save_path="parameters.png" if save else None)
        if show:
            plt.show()
        return figs


if __name__ == "__main__":

    # Forward problem: learn the trajectory on [0, 1] from (1, 1, 1).
    # Tuned settings (see results/all_results.csv): RK4 data at dt = 1e-4,
    # Adam at learning rate 1e-2 for 10,000 epochs, then 20,000 L-BFGS steps.
    solver = Lorenz_PINN_solver(T=1.0, dt=[.0001], method="rk4", optimizer="adam",
                                learning_rate=1e-2, lbfgs_iters=20000)
    pred_df = solver.pinn_pipeline(dt=.0001, n_sample=200, epochs=10000)

    # Visuals: every plot at once (also saved to ./plots)
    vis = Lorenz_Visuals(solver, save_dir="plots")
    vis.plot_all()

    # Individual plots, e.g. zoom in on the first 4000 time steps (t = 0 to 0.4)
    # vis.plot_time_series(n_steps=4000)
    # vis.plot_errors(n_steps=4000)
    # vis.animate_prediction(save_path="prediction.gif")

    # Inverse problem: start from wrong guesses and learn sigma, rho, beta
    # inverse_solver = Lorenz_PINN_solver(T=1.0, dt=[.0001], inverse=True)
    # inverse_solver.pinn_pipeline(dt=.0001, initial_params=(5., 20., 2.))
    # Lorenz_Visuals(inverse_solver).plot_parameters()
