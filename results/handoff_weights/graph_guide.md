# What happens to the weights at each handoff: warm start vs fresh weights

Two time-marching runs, identical except for warm start: T = 5, 0.25-unit windows (20 windows), data everywhere, seed 99, 10,000 Adam epochs and up to 5,000 L-BFGS steps per window. Both runs share window 1 exactly, so every difference after that comes from the handoff.

**The short version:** warm start passes the weights on, and they get bigger in every window. As they grow, more and more tanh units get pushed into their flat ends, where they stop passing gradient. By t = 4 about half of the network has stopped responding to training. At the window starting at t = 4.25 it stopped training altogether. Fresh weights reset to the same small size every window and never build up. The error at the handoffs gives no warning before the failure; the weights do.

Made with `scripts/handoff_analysis.py` from runs saved by `scripts/time_marching.py --save-dir`. Numbers per window are in `handoff_summary.csv`.

---

## 1. Errors at the handoffs (`handoff_errors.png`)

**What it shows.** Top: the 3D distance between the PINN and the RK4 reference along the path. Bottom: the error each window hands to the next one (its end point, which becomes the next start point). Dotted lines are handoffs.

**What to look for.** Whether errors build up handoff by handoff, or jump.

**What happened.** Both runs handed over errors between 1e-7 and 1e-5 at every handoff up to t = 4.25. Warm start did not drift upward at the handoffs. It held, then jumped to 2.8 at t = 4.5 in one window. So the handoff error is a poor early warning. The trouble was building up inside the network, as the next graphs show.

## 2. Losses per window (`handoff_losses.png`)

**What it shows.** The training loss of each window at three moments: epoch 1 (how good the starting guess is), after 10,000 Adam epochs, and after L-BFGS.

**What to look for.** Whether the copied weights give a better start, and whether each window trains as well as the last.

**What happened.**
- **Epoch 1:** warm starts were about 3 times worse in every window (10 to 17 against 3 to 6). The copied network is a worse starting guess than random weights.
- **After Adam:** fresh stayed between 1e-7 and 1e-6 in every window. Warm crept up from 1e-6 to 1.5e-4 by t = 4, about 100 times worse. This is the clearest early warning in the losses.
- **After L-BFGS:** both reached about 1e-9, so L-BFGS covered up the problem until the window at t = 4.25, which stayed at 2.4.

## 3. The starting guess at a handoff (`starting_guess.png`)

**What it shows.** For three windows: the reference path (black), what the copied warm network predicts before any training (pink), and what the fresh network predicts before any training (green). Both start at the same point because of the hard initial condition.

**What to look for.** Whether the copied network's curve looks anything like the path it now has to learn.

**What happened.** It doesn't. The warm guess is the previous window's curve redrawn on the new window: full-size, with its own bends, and often going the wrong way (window 10: y should rise, the guess falls). The fresh guess is nearly flat, which is wrong too, but less wrong. This is why epoch 1 starts worse with warm start. The network sees only local time on [-1, 1], so nothing tells it the new window comes after the old one.

## 4. Weight size (`weight_size.png`)

**What it shows.** The size (norm) of each layer's weights at the end of each window, and of all 12,803 weights together.

**What to look for.** Steady growth means each window adds to the weights and nothing takes it away.

**What happened.** This is the main finding. Warm weights grew in every single window: the total went from 16.5 to 61.5 by t = 4, then to 100 after the failure. Hidden layers 2 to 4 grew 3 to 4 times. Fresh weights finished every window at about the same size (16 to 18). The one exception was a hard window at t = 0.5, which went to 32, and the next fresh window reset it. Nothing in the code shrinks the weights, so under warm start each window's training is stacked on top of all the earlier ones.

## 5. Health of the hidden units (`unit_health.png`)

**What it shows.** Top row: the share of hidden units sitting in the flat ends of tanh (input bigger than 2 in size), measured over the whole window. There the slope is under 0.07, so very little gradient gets through. Bottom row: how many different shapes the 64 units in a layer actually make (the rank of the layer's outputs). Because the input is a single number (time), this is small for both runs. A drop means units are copying each other or have gone flat.

**What to look for.** Rising saturation means the network is losing usable units.

**What happened.**
- **Layer 2:** warm went from 0% saturated to 88% by t = 4. Fresh stayed under 1%, apart from the hard window at t = 0.5.
- **Layers 3 and 4:** warm sat at 85% to 97% from t = 0.5 on. Fresh moved between 5% and 77% from window to window, apart from the hard window at t = 0.5.
- **The failed window:** in the window at t = 4.25, layers 3 and 4 were 100% saturated and made just one shape. Every unit was stuck at +1 or -1, so the network could only output a near-constant, which is the "almost straight line" seen in failed windows before.
- **Shapes per layer:** warm settled at 2 to 3 in layers 3 and 4, against 4 to 8 for fresh.

Growing weights (graph 4) push the tanh inputs past 2, which is exactly this saturation.

## 6. How the weights move (`weight_movement.png`)

**What it shows.** Left: how far the weights travel from the start to the end of each window. Middle: how much of that trip is done after 1, 10, 100, 1,000 and 2,000 Adam steps. Right: the size of Adam's very first step in each window, as a share of a full step.

The right panel needs one fact about Adam. Its first step moves every weight by about the learning rate (0.01), unless that weight's gradient is close to zero. So the first step's length tells you roughly how many weights are still getting gradient. If it is 0.5 of a full step, roughly a quarter of the weights are moving and the rest are stuck (a rough reading).

**What to look for.** Whether warm start saves travel, and how many weights still respond to training.

**What happened.**
- **Left:** warm windows travelled at least as far as fresh ones (about 10 per window for fresh, 10 to 20 for warm). The copied weights were not closer to the answer, which is why warm start saved no time.
- **Middle:** both runs do most of their moving in the first 100 Adam steps, while the physics is still switched off and the learning rate is at its largest. Whatever the copied weights knew is mostly overwritten in those first steps.
- **Right:** fresh took a full first step in every window. Warm's first step shrank steadily, from 1.0 to 0.5 at t = 4 and 0.14 after the failure. That points to about three quarters of the weights getting almost no gradient by t = 4, the same saturation seen in graph 5.

---

## What this means

The earlier hypothesis holds up in this run (one seed). Warm start doesn't fail because of the errors it passes on. It fails because the copied weights keep growing, the growth saturates the tanh units, and the network slowly loses the units it can train. Fresh weights avoid that by starting small every time.

Cheap things to try if warm start is still wanted: shrink the copied weights toward zero before each window (for example times 0.3, plus a little noise), copy only the hidden layers, or add a small penalty on weight size. Graphs 4 to 6 would show right away whether a fix stops the build-up.
