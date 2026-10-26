# The Lorenz PINN code, explained

This guide walks through `scripts/lorenz_pinn.py` from top to bottom. `scripts/lorenz_pinn_full.py` is the same code plus a `Lorenz_Visuals` class at the end that only draws plots, so once you understand this file you understand both.

## The big picture in one paragraph

The code does three things. First, it computes the "right answer" for the Lorenz system the old-fashioned way, by stepping forward in tiny time steps (Euler's method). Second, it trains a neural network that takes a time `t` and outputs a guess for `(x, y, z)` at that time. Third, it measures how far the network's guesses are from the right answer. The "physics-informed" part is that the network is graded not only on matching some known points but also on obeying the Lorenz equations everywhere in between.

## What the Lorenz system is

It is three equations describing how three numbers `x`, `y`, `z` change over time:

```
dx/dt = sigma * (y - x)
dy/dt = x * (rho - z) - y        (the code writes this as rho*x - y - x*z, same thing)
dz/dt = x * y - beta * z
```

`sigma = 10`, `rho = 28`, `beta = 8/3` are the classic constants. "dx/dt" just means "how fast x is changing right now". If you know where you are now, the equations tell you which way you're moving, so you can step forward a little bit, recompute, step again, and trace out the whole path. The system is chaotic: tiny differences in the start point grow into huge differences later. That is why the code only trains on a short window, `T = 1.0`.

## What a PINN is

A neural network here is just a flexible function: you feed in `t`, it spits out three numbers. Training means nudging its internal weights so the outputs get better. "Better" is defined by a **loss**, a single number that's big when the network is wrong and small when it's right. Training = repeatedly lowering the loss.

A normal network's loss only says "match these known data points". A **Physics-Informed** Neural Network adds a second grade: "at any time t, the slope of your curve should equal what the Lorenz equations say it should be". TensorFlow can compute the slope of the network's output with respect to `t` automatically, so we can check that at thousands of random times, even where we have no data. The network is pushed to be both close to the data and physically consistent.

## Walking through the file

### `linear_space` (top of file)

A homemade version of `np.linspace`. **Nothing in this file calls it**, so you can ignore it.

### Class `Lorenz`: making the reference answer

Think of this class as "the textbook solver".

- `__init__` stores the constants (`sigma`, `rho`, `beta`), the start point `(x0, y0, z0) = (1, 1, 1)`, the end time `T`, and a list of step sizes `dt`.
- `get_ic_list(dt)` is the heart of it. It runs Euler's method: start at `(1,1,1)`, compute `dx, dy, dz` from the equations, move a tiny step (`x += dt * dx`, etc.), and repeat `T/dt` times. Every position and time is saved into `x_list`, `y_list`, `z_list`, `t_list`. With `dt = 0.0001` and `T = 1`, that's 10,000 steps.
- `get_lorenz_system()` runs that solver once for **each** `dt` in the list and stores each result.
- `get_dataframe()` stacks all those runs into one pandas table, `lorenz_df`, with columns `t, x_pos, y_pos, z_pos, dt` (plus the constants). The `dt` column tells you which run a row came from.
- `lorenz_pipeline()` just calls the two above in order.

Why several `dt`s? Smaller steps give a more accurate path. Comparing them shows how much the "right answer" itself depends on step size.

### Class `Lorenz_PINN_solver(Lorenz)`: the neural network part

It inherits from `Lorenz`, so it has all the solver stuff above plus the network. The sections are numbered in the code with comment banners, and I follow those numbers.

**`__init__`** stores the settings:
- `hidden_layers=4, units=64`: the network is 4 layers of 64 "neurons" each.
- `learning_rate`: how big each nudge to the weights is.
- `n_colloc=2000`: how many random times to check the physics at, each training step ("collocation points").
- `loss_weights=(1, 1, 10)`: how much to care about data, physics, and the start point respectively.
- `warmup_epochs`: how long to train on data alone before turning the physics on (explained in section 5).
- `inverse`: a second mode where the network also tries to *discover* sigma, rho and beta (explained at the end).

**1. Sampling (`sample_system`, `get_samples`, `sample_pipeline`)**

You don't give the network all 10,000 reference points, only a sample, so the physics has to fill the gaps. `sample_system` cuts the time range `[0, T]` into 200 equal slices and picks one random row from each slice. That guarantees the 200 points are spread evenly with no big holes. `sample_pipeline` runs the Euler solver and then the sampling.

**2. `prepare_training_data`**

Converts the 200 sampled rows into TensorFlow tensors:
- `t_data`: the 200 times; `state_data`: the matching `(x, y, z)` values.
- `t_ic`, `state_ic`: the single point `t = 0 → (1, 1, 1)`, the initial condition.
- `state_mean`, `state_std`: the average and spread of x, y and z. `z` gets up to about 45 while `x` stays smaller, so without rescaling, errors in `z` would dominate. These are used to put everything on a similar scale.

**3. `build_network` and `predict_state`**

`build_network` creates the network: 1 input (time) → 4 layers of 64 with `tanh` → 3 outputs. `tanh` is used instead of the more common ReLU because the physics check takes the slope of the output, and `tanh` is smooth while ReLU has sharp corners. It also makes `sigma_tf, rho_tf, beta_tf` as TensorFlow variables (frozen unless `inverse=True`), creates the Adam optimizer (the thing that nudges the weights), and wraps the training step with `tf.function` so it runs fast.

`predict_state(t)` is the function you actually call to ask the network "where am I at time t?". It squeezes `t` from `[0, T]` into `[-1, 1]` (networks like small inputs), runs the network, and then un-scales the output with `std` and `mean` to get real `x, y, z` values.

**4. `physics_residual(t)`**: the "physics-informed" bit

This is the part that's hardest to read, so slowly:
1. `tf.GradientTape` is TensorFlow's way of recording calculations so it can compute derivatives afterward. `tape.watch(t)` says "I'll want derivatives with respect to t".
2. Inside the tape it calls `predict_state(t)` and splits the result into `x`, `y`, `z`.
3. `tape.gradient(x, t)` gives `dx/dt`, the slope of the network's x-curve. Same for y and z.
4. It computes what the slopes *should* be from the Lorenz equations: `f_x, f_y, f_z`.
5. The **residual** is the difference: `dx/dt - f_x`, etc. If the network obeys the physics perfectly, the residual is zero.
6. The last line rescales the residual so it's on the same scale as the data error.

**5. Loss and training (`compute_loss`, `_train_step`, `train`)**

`compute_loss` builds three grades and adds them up:
- **Data loss**: how far the network is from the 200 sampled points (mean squared error, rescaled).
- **Physics loss**: the squared residual at 2000 brand-new random times, drawn fresh every step.
- **Initial-condition loss**: how far the network is from `(1,1,1)` at `t = 0`.
- **Total** = `w_data·data + w_phys·physics + w_ic·ic`.

`_train_step` does one round of learning: compute the total loss inside a second `GradientTape`, ask for the gradient of the loss with respect to every network weight ("which way should each weight move to lower the loss?"), and let Adam apply that nudge. Note there are two tapes in play: the inner one in `physics_residual` gives slopes with respect to time, and this outer one gives slopes with respect to the weights.

`train(epochs)` calls the step 10,000 times. Each "epoch" here is one step. The interesting part is the **warm-up**:
- Epochs 1–1000: physics weight is 0, so it learns from data only.
- Epochs 1000–2000: physics weight ramps up from 0 to its full value.
- After that: everything at full strength.

The comment explains why: if physics is on at full strength from the start, the network tends to find a smooth curve that roughly obeys the equations but ignores the data. Getting the rough shape from data first avoids that trap. Every step's loss values are saved in `loss_history` (and `loss_df` at the end), which is what the plotting class uses.

**6. `predict`: grading the result**

Takes the full Euler run (all 10,000 points for the `dt` you trained on), asks the network for its prediction at every one of those times, and puts both side by side in `pred_df`. Then it prints the **relative L2 error** for each of x, y, z:

```
error = size of (network - euler) / size of (euler)
```

This is the number your goal refers to: you want it below `1e-4`.

**7. `pinn_pipeline`**: one call to run everything in order: Euler solve → sample → tensors → build network → train → predict.

### The bottom of the file (`if __name__ == "__main__":`)

This is what runs when you do `python scripts/lorenz_pinn.py`. It builds a solver over `[0, 1]` with Euler step sizes `0.001` and `0.0001`, trains on the `0.0001` run with 200 sampled points for 10,000 steps, and prints the errors. The commented-out lines show the **inverse problem**: start with wrong guesses for the constants (`sigma=5, rho=20, beta=2`), set `inverse=True`, and the optimizer adjusts those constants along with the weights to see if it can recover 10, 28, 8/3 from the data.

## The plotting class (only in `lorenz_pinn_full.py`)

`Lorenz_Visuals(solver)` takes a trained solver and draws: the reference 3D butterfly (`plot_reference_3d`), the network's version (`plot_predicted_3d`), x/y/z over time (`plot_time_series`), an animation, the error over time on a log scale (`plot_errors`), each loss term over training with the warm-up shaded (`plot_losses`, `plot_loss_minima`), and, in inverse mode, how sigma/rho/beta moved during training (`plot_parameters`). `plot_all` runs them all. None of it changes the training.

## Two things worth knowing for your 1e-4 goal

The first point uses numbers measured in another thread; the second is from reading the code.

1. **The "right answer" is itself approximate.** The network is graded against Euler's method, which has its own error that shrinks roughly in proportion to `dt`. Another thread in this project measured it: Euler at `dt = 1e-4` compared with a much more accurate RK4 solver at `dt = 1e-5` over `T = 1` was off by up to about `7.6e-2`, with a relative L2 error of about `3e-3`. That is well above your `1e-4` target, so "within 1e-4 of Euler" does not mean "within 1e-4 of the true solution". To reach the goal against the true path, the reference data needs a more accurate solver (such as RK4) or a much smaller `dt`.
2. **The network is trained in 32-bit floats** (`tf.float32`), which carry about 7 significant digits. That's enough for 1e-4, but it doesn't leave a lot of headroom.
