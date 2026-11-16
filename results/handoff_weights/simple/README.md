# Weight size at each handoff: one number per handoff

Same runs as `../handoff_errors.png`: T = 5, 20 windows of 0.25 time units (handoffs at t = 0.25, 0.5, ..., 5), data everywhere, seed 99, 10,000 Adam epochs and up to 5,000 L-BFGS steps per window. Nothing was retrained; the numbers come from the saved weights in `../runs/`.

**The number.** At each handoff, take every weight and bias in the 4 hidden layers of 64 units (12,736 numbers; the output layer is left out) from the window's final network, and compute their RMS. Divide by the RMS of the random network the run started from (0.123). So 1.0 means "the same size as a newly made network" and 4.0 means "four times bigger". Both graphs use the same y-axis so they can be compared directly.

## `weight_rms_both.png` (both on one graph)

The two graphs below drawn on the same axes: warm start in pink, no warm start in green. The red X's and the circled last good handoff belong to the warm run. The lines stay together until t = 0.75 and then split apart for good.

## `weight_rms_warm.png` (warm start)

The weights grow at every handoff and never come back down: 2.09 at t = 1, 2.81 at t = 2, 3.33 at t = 3, 4.13 at t = 4, and 4.35 at t = 4.25, the **last good handoff** (circled). The next window (4.25 to 4.5) went off the rails, ending 2.8 away from the reference; those windows are the red X's, where the size jumps to 6.45 by t = 5.

## `weight_rms_fresh.png` (no warm start)

Every window starts again at 1.0, and training takes it to about 1.2 to 1.3 at each handoff, every time. The one spike is the window from 0.5 to 0.75 (2.30), a hard stretch of the path; the next fresh network reset it. No window failed.

## Takeaway

Warm start keeps stacking each window's training on top of the last, so the hidden weights were over 4 times their starting size just before it failed. Fresh networks stay at the same small size the whole way. The handoff errors (`../handoff_errors.png`) gave no warning before t = 4.5; this number climbed steadily the whole time.

Made with `scripts/handoff_weight_rms.py`. Per-handoff values are in `weight_rms.csv`.
