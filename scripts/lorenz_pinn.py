import numpy as np
import pandas as pd
import tensorflow as tf


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
                 T=1.0, x0=1., y0=1., z0=1.):

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

            dx = self.sigma * (y - x)
            dy = self.rho * x - y - x * z
            dz = x * y - self.beta * z

            x += dt * dx
            y += dt * dy
            z += dt * dz

            self.x_list.append(x)
            self.y_list.append(y)
            self.z_list.append(z)
            self.t_list.append((i + 1) * dt)  # step count * dt avoids drift from repeated adding

    def get_lorenz_system(self):

        for dt in self.dt_list:

            self.get_ic_list(dt=dt)
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

    def __init__(self, hidden_layers=4, units=64, learning_rate=1e-3,
                 n_colloc=2000, loss_weights=(1.0, 1.0, 10.0),
                 warmup_epochs=1000, inverse=False, seed=99, **lorenz_kwargs):
        """
        hidden_layers, units : size of the network
        n_colloc             : physics (collocation) points drawn each epoch
        loss_weights         : (data, physics, initial condition)
        warmup_epochs        : epochs of data-only training, after which the
                               physics weight ramps up over another warmup_epochs
        inverse              : if True, sigma / rho / beta become trainable
        lorenz_kwargs        : anything Lorenz takes (sigma, T, dt, x0, ...)
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

        # The physics weight changes during training, so it is a tf.Variable.
        # A plain Python float would get frozen into the tf.function graph.
        self.w_phys = tf.Variable(0.0, dtype=tf.float32, trainable=False)
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
        
        time_bins = pd.cut(current_df["t"], bins=n_sample) # weighted sample for more [.5 - 1]
        sample = (current_df.groupby(time_bins, observed=True)
                            .sample(n=1, random_state=self.seed)
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
        self.t_data = tf.constant(sample[["t"]].values, dtype=tf.float32)
        self.state_data = tf.constant(sample[cols].values, dtype=tf.float32)

        # Initial condition: a single point at t = 0
        self.t_ic = tf.zeros((1, 1), dtype=tf.float32)
        self.state_ic = tf.constant([[self.x0, self.y0, self.z0]], dtype=tf.float32)

        # Output scaling. The network predicts roughly unit-sized numbers and we
        # map them back with mean/std. Without this, z (up to ~45) would dominate.
        self.state_mean = tf.constant(sample[cols].mean().values.reshape(1, 3), dtype=tf.float32)
        self.state_std = tf.constant(sample[cols].std().values.reshape(1, 3), dtype=tf.float32)

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
            h = tf.keras.layers.Dense(self.units, activation="tanh",
                                      kernel_initializer="glorot_normal")(h)
        outputs = tf.keras.layers.Dense(3)(h)  # x, y, z (scaled)
        self.model = tf.keras.Model(inputs, outputs)

        # sigma, rho, beta as tf.Variables. With inverse=False they are frozen
        # constants; with inverse=True the optimizer learns them from the data.
        if initial_params is None:
            initial_params = (self.sigma, self.rho, self.beta)
        s0, r0, b0 = initial_params
        self.sigma_tf = tf.Variable(s0, dtype=tf.float32, trainable=self.inverse)
        self.rho_tf = tf.Variable(r0, dtype=tf.float32, trainable=self.inverse)
        self.beta_tf = tf.Variable(b0, dtype=tf.float32, trainable=self.inverse)

        self.train_vars = list(self.model.trainable_variables)
        if self.inverse:
            self.train_vars += [self.sigma_tf, self.rho_tf, self.beta_tf]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Wrap the train step in tf.function here, after the data is prepared,
        # so the compiled graph uses this run's tensors.
        self.train_step = tf.function(self._train_step)

    def predict_state(self, t):
        """Physical time in, physical (x, y, z) out.

        Scaling happens inside this function, so when GradientTape
        differentiates it with respect to t, the chain rule through the
        scaling is handled automatically.
        """
        t_norm = 2.0 * t / self.T - 1.0  # [0, T] -> [-1, 1]
        out = self.model(t_norm)
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

    def compute_loss(self):

        # Data loss: match the sampled Euler points
        pred_data = self.predict_state(self.t_data)
        loss_data = tf.reduce_mean(tf.square((pred_data - self.state_data) / self.state_std))

        # Physics loss: fresh random collocation times every step
        t_colloc = tf.random.uniform((self.n_colloc, 1), 0.0, self.T, dtype=tf.float32)
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

    def train(self, epochs=10000, print_every=1000):

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

        self.loss_df = pd.DataFrame(self.loss_history)

    # ------------------------------------------------------------------
    # 6. Predict and compare with the full Euler trajectory
    # ------------------------------------------------------------------

    def predict(self, every=1):
        """Evaluate the PINN at the times of the full Euler run it was trained on.

        every : keep every k-th row (useful for dt = 1e-6, which has a million rows).
        """
        ref = (self.lorenz_df.loc[self.lorenz_df["dt"] == self.train_dt]
                             .iloc[::every]
                             .reset_index(drop=True))

        t = tf.constant(ref[["t"]].values, dtype=tf.float32)
        pred = self.predict_state(t).numpy()

        self.pred_df = ref[["t", "x_pos", "y_pos", "z_pos"]].copy()
        self.pred_df["x_pinn"] = pred[:, 0]
        self.pred_df["y_pinn"] = pred[:, 1]
        self.pred_df["z_pinn"] = pred[:, 2]

        # Relative L2 error per variable: ||pinn - euler|| / ||euler||
        for v in ["x", "y", "z"]:
            true = self.pred_df[f"{v}_pos"].values
            err = np.linalg.norm(self.pred_df[f"{v}_pinn"].values - true) / np.linalg.norm(true)
            print(f"relative L2 error {v}: {err:.3e}")

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


if __name__ == "__main__":

    # Forward problem: learn the trajectory on [0, 1] from (1, 1, 1)
    solver = Lorenz_PINN_solver(T=1.0, dt=[.001, .0001])
    pred_df = solver.pinn_pipeline(dt=.0001, n_sample=200, epochs=10000)

    # Inverse problem: start from wrong guesses and learn sigma, rho, beta
    # inverse_solver = Lorenz_PINN_solver(T=1.0, dt=[.0001], inverse=True)
    # inverse_solver.pinn_pipeline(dt=.0001, initial_params=(5., 20., 2.))
