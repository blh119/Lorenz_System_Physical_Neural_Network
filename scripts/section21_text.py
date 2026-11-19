"""Words and tables for report Section 21 (used by pct_report_section.py)."""
import re

import numpy as np
import pandas as pd

SECTION_TITLE = "Filling in the data-needed table"


def story_for(base, ex, filled, figs, k):
    P, h1, h2, body, cap = k["Paragraph"], k["h1"], k["h2"], k["body"], k["cap"]
    B, box, grid, sci, Image, Spacer, inch = k["B"], k["box"], k["grid"], k["sci"], k["Image"], k["Spacer"], k["inch"]
    KeepTogether, PageBreak = k["KeepTogether"], k["PageBreak"]
    s = []

    def r(name):
        return ex.loc[ex["name"] == name].iloc[0]

    s += [P("21. " + SECTION_TITLE, h1)]
    s += [box("In plain words",
              "Section 18 left 13 of the 30 cells in its table empty (\"none\"). Brian asked to fill every cell "
              "possible, using as much data as it takes, up to all of the reference points, and to report any cell "
              "that still can't be filled as a failed run, with the reason. Two things were tried, in order. "
              "<b>More data:</b> 50% and 100% of the points at every T, with the Section 18 settings. That filled "
              "<b>no</b> new cells. <b>Longer training:</b> for the cells still empty, L-BFGS was allowed 20,000 to "
              "180,000 steps instead of 5,000, and T = 3 was retried with other seeds. That filled 9 of the 13. "
              "Four cells still fail: 10<super>-6</super> at T = 2.5, and 10<super>-4</super>, 10<super>-5</super> "
              "and 10<super>-6</super> at T = 3.")]
    s += [Spacer(1, 8)]

    s += [P("21.1 How the runs were set up", h2)]
    s += [P("Everything is the same as Section 18.1: one 4 × 64 tanh network on all of [0, T], float64, RK4 "
            "reference at dt = 10<super>-4</super>, Adam at 0.01 with cosine decay for 10,000 epochs, then L-BFGS; "
            "2,000 physics points per unit of time for Adam and 4,000 for L-BFGS; data spread evenly; seed 99 "
            "unless stated. The script was rebuilt as <font face='Courier'>scripts/pct_sweep.py</font>. As a check, "
            "it re-ran T = 1 at 0.1% and got a relative L2 of 3.458 × 10<super>-6</super>, the same as Section 18 "
            "to every printed digit.", body)]
    s += B(["<b>100% of the data</b> means every reference point: 10,001 at T = 1 up to 30,001 at T = 3.",
            "<b>L-BFGS cap.</b> Section 18 stopped L-BFGS at 5,000 steps. Here it was allowed 20,000, 50,000, "
            "60,000, 100,000, or up to 250,000 (stopped earlier). One detail: SciPy also stops L-BFGS after 15,000 loss evaluations, so the runs "
            "marked \"20k\" actually stopped at about 13,700 steps. The longer runs raised that limit too, and "
            "ran their full number of steps.",
            "<b>Checkpoints.</b> The cloud machine restarted once, part-way through, and three long runs were lost. "
            "They were started again from scratch, with the weights saved every 5,000 L-BFGS steps. "
            "That also records the error as training goes (Figure 21.2).",
            "<b>A cell is filled</b> by the smallest % whose relative L2 over [0, T] is at or under the level. "
            "Runs with the original settings are used first. A cell that needed longer L-BFGS or another seed "
            "says so (green in Figure 21.3)."])

    # ---------------- 21.2 more data
    s += [P("21.2 Step 1: more data (50% and 100%)", h2)]
    rows = [["T", "Ref. pts", "Best in Section 18 (0.01% to 20%)", "50%", "100%"]]
    for T in [1, 1.5, 2, 2.5, 3]:
        b = base[base["T"] == T].dropna(subset=["rel_l2"]).sort_values("rel_l2").iloc[0]
        r50 = ex[(ex["T"] == T) & (ex["pct_reference"] == 50) & (ex["lbfgs_cap"] == 5000) & (ex["seed"] == 99)]
        r100 = ex[(ex["T"] == T) & (ex["pct_reference"] == 100) & (ex["lbfgs_cap"] == 5000) & (ex["seed"] == 99)]
        rows.append([f"{T:g}", f"{int(b.n_reference):,}", f"{sci(b.rel_l2)} at {b.pct_reference:g}%",
                     f"{sci(r50.rel_l2.iloc[0])} ({int(r50.n_data.iloc[0]):,} pts)",
                     f"{sci(r100.rel_l2.iloc[0])} ({int(r100.n_data.iloc[0]):,} pts)"])
    s += [grid(rows, [0.45 * inch, 0.75 * inch, 2.1 * inch, 1.6 * inch, 1.6 * inch])]
    s += [P("Relative L2 over [0, T]. Seed 99, 5,000 L-BFGS steps, as in Section 18.", cap)]
    s += [Image(figs["data"], width=6.5 * inch, height=6.5 * inch * 4.6 / 9)]
    s += [P("Figure 21.1. Section 18's sweep with the new 50% and 100% points added (shaded). Original settings "
            "throughout.", cap)]
    s += B(["<b>Up to T = 2, more data did not make it better.</b> At T = 1, 100% of the points gave "
            f"{sci(r('T1_pct100').rel_l2)}, about the same as 10 points. At T = 1.5 and 2, 50% and 100% came out "
            f"worse than the best small amounts ({sci(r('T1.5_pct100').rel_l2)} and {sci(r('T2_pct100').rel_l2)} "
            "at 100%).",
            "<b>At T = 2.5 and 3, 50% and 100% failed</b>, at 51% to 55% error, like 2% to 20% did in Section 18. "
            "The loss after the 10,000 Adam epochs was still about 2.6, so these runs were stuck before L-BFGS "
            "even started. L-BFGS only polishes a solution that is already close, so it can't rescue them.",
            "So step 1 filled no new cells. The empty cells are limited by training, not by data."])

    # ---------------- 21.3 longer training
    s += [P("21.3 Step 2: longer training and other seeds", h2)]
    order = ["T1_pct2_lbfgs20000", "T1_pct0.1_lbfgs20000", "T1.5_pct2_lbfgs60000_maxfun120000",
             "T1.5_pct0.25_lbfgs60000_maxfun120000", "T2_pct2_lbfgs20000", "T2_pct2_lbfgs60000_maxfun120000",
             "T2_pct0.25_lbfgs60000_maxfun120000", "T2.5_pct1_lbfgs20000", "T2.5_pct1_lbfgs50000_maxfun100000",
             "T2.5_pct1_lbfgs100000_maxfun200000", "T3_pct10_lbfgs20000", "T3_pct10_lbfgs60000_maxfun120000",
             "T2.5_pct1_lbfgs250000_maxfun500000", "T3_pct10_lbfgs200000_maxfun400000",
             "T3_pct10_s7", "T3_pct10_s123", "T3_pct100_s7"]
    rows = [["T", "% (pts)", "Seed", "L-BFGS steps run", "Relative L2", "Hours", "Fills"]]
    fills = {"T1_pct2_lbfgs20000": "10<super>-6</super> (later beaten by 0.1%)",
             "T1_pct0.1_lbfgs20000": "T = 1, 10<super>-6</super>",
             "T1.5_pct2_lbfgs60000_maxfun120000": "10<super>-6</super> (later beaten by 0.25%)",
             "T1.5_pct0.25_lbfgs60000_maxfun120000": "T = 1.5, 10<super>-6</super>",
             "T2_pct2_lbfgs20000": "10<super>-5</super> (later beaten by 0.25%)",
             "T2_pct2_lbfgs60000_maxfun120000": "T = 2, 10<super>-6</super>",
             "T2_pct0.25_lbfgs60000_maxfun120000": "T = 2, 10<super>-5</super> (not 10<super>-6</super>)",
             "T2.5_pct1_lbfgs20000": "T = 2.5, 10<super>-4</super>",
             "T2.5_pct1_lbfgs50000_maxfun100000": "none (just missed 10<super>-5</super>)",
             "T2.5_pct1_lbfgs100000_maxfun200000": "T = 2.5, 10<super>-5</super>",
             "T3_pct10_lbfgs20000": "none",
             "T3_pct10_lbfgs60000_maxfun120000": "T = 3, 10<super>-3</super>",
             "T2.5_pct1_lbfgs250000_maxfun500000": "none: stalled near 3.9 × 10<super>-6</super> (stopped at 180,000)",
             "T3_pct10_lbfgs200000_maxfun400000": "none: stalled near 5 × 10<super>-4</super> (stopped at 140,000)",
             "T3_pct10_s7": "none (stuck)", "T3_pct10_s123": "none (stuck)", "T3_pct100_s7": "none (stuck)"}
    for n in order:
        if n not in set(ex["name"]):
            continue
        x = r(n)
        rows.append([f"{x['T']:g}", f"{x.pct_reference:g}% ({int(x.n_data):,})", str(int(x.seed)),
                     f"{int(x.lbfgs_steps):,}", sci(x.rel_l2),
                     "–" if pd.isna(x.seconds) else f"{x.seconds / 3600:.1f}", fills[n]])
    s += [grid(rows, [0.4 * inch, 0.9 * inch, 0.45 * inch, 1.0 * inch, 1.05 * inch, 0.5 * inch, 2.2 * inch])]
    s += [P("Every run after step 1. \"L-BFGS steps run\" is where L-BFGS actually stopped; runs with a 20,000 cap "
            "hit SciPy's 15,000-evaluation limit first. Hours are on one CPU core, with four runs at a time. The two "
            "longest runs continued the 100,000- and 60,000-step runs from their checkpoints and were stopped at their "
            "last checkpoint (no hours shown: the machine restarted twice under them).", cap)]
    s += [Image(figs["steps"], width=6.5 * inch, height=6.5 * inch * 4.4 / 9)]
    s += [P("Figure 21.2. Relative L2 at each checkpoint (every 5,000 L-BFGS steps) for the long runs that saved "
            "checkpoints. Dotted lines: 10<super>-6</super> to 10<super>-2</super>; dashed: the 10<super>-4</super> "
            "target. The T = 2.5 and T = 3 lines continue past 100,000 and 60,000 steps in the follow-up runs.", cap)]
    s += B(["<b>10<super>-6</super> is reachable up to T = 2.</b> It needed about 14,000 L-BFGS steps at T = 1, "
            "and 60,000 at T = 1.5 and 2, with very little data (10 points at T = 1, 38 at T = 1.5, 400 at T = 2). "
            "At T = 2, 0.25% (50 points) got to 1.9 × 10<super>-6</super> in the same 60,000 steps but not under "
            "10<super>-6</super>.",
            "<b>T = 2.5 improves slowly, then stalls.</b> At 1% of the points, the error went from 4.8 × 10<super>-5</super> "
            "(about 14,000 steps) to 1.0 × 10<super>-5</super> (50,000) to 5.3 × 10<super>-6</super> (100,000): "
            "then 3.9 × 10<super>-6</super> at 180,000, where it flattened out. 10<super>-6</super> would need about "
            "4 times less error, and the last 30,000 steps gained almost nothing.",
            "<b>T = 3 gets under 10<super>-3</super>, then stalls.</b> 10% of the points went from 7.1 × "
            "10<super>-3</super> (5,000 steps) to 2.0 × 10<super>-3</super> (14,000) to 6.7 × 10<super>-4</super> "
            "(60,000) and 5.2 × 10<super>-4</super> (125,000), then wandered between 5.2 and 5.8 × 10<super>-4</super> "
            "up to 140,000 steps. 10<super>-4</super> would need about 5 times less error.",
            "<b>Other seeds did not help T = 3.</b> Seeds 7 and 123 at 10%, and seed 7 at 100%, all got stuck at "
            "51% to 96% error. The one working T = 3 run (seed 99, 10%) is the lucky one, as Section 18 suspected."])

    # ---------------- 21.4 table
    s += [KeepTogether([P("21.4 The filled-in table", h2),
                        Image(figs["table"], width=6.7 * inch, height=6.7 * inch * 931 / 1800)])]
    s += [P("Figure 21.3. Smallest tested % of the reference points that reaches each level of relative L2 over "
            "[0, T]. Blue: Section 18 settings. Green: needed a longer L-BFGS cap (shown). Red: no run reached it; "
            "the best error is shown. Also saved as results/pct_sweep/pct_needed_table_filled.xlsx.", cap)]

    # ---------------- 21.5 failed
    s += [P("21.5 The cells that failed, and why", h2)]
    fail = filled[filled["status"] == "failed"].sort_values(["T", "level"], ascending=[True, True])
    rows = [["T", "Level", "Best run", "Why it failed"]]
    for x in fail.itertuples():
        rows.append([f"{x.T:g}", x.level.replace("≤ 1e-", "&lt;= 10<super>-") + "</super>", f"{sci(x.rel_l2)}<br/>({x.run})",
                     re.sub(r"(\d(?:\.\d+)?)e-0?(\d)", r"\1 × 10<super>-\2</super>", x.reason.replace(" 1e-4", " 10<super>-4</super>"))])
    s += [grid(rows, [0.4 * inch, 0.6 * inch, 1.5 * inch, 4.1 * inch])]
    s += [Spacer(1, 6)]
    s += [box("Key takeaway",
              "Data was never what was missing. Every newly filled cell used 10% of the points or less (2% or "
              "less up to T = 2.5), and going to 50% or 100% of the points filled nothing. What filled the cells was letting L-BFGS "
              "run longer: 10<super>-6</super> now holds out to T = 2, 10<super>-5</super> to T = 2.5 and the "
              "10<super>-4</super> target to T = 2.5. One network on [0, 3] still fails, because Adam usually "
              "doesn't find the right path at all there, whatever the data or the seed. For [0, 3] and beyond, "
              "time-marching (Sections 16, 17 and 20) remains the way to go.")]

    s += [P("21.6 Limits and next steps", h2)]
    s += B(["One run per setting, mostly seed 99. The long-L-BFGS cells were not repeated with other seeds.",
            "\"Smallest %\" in a green cell is the smallest % that was tried with long L-BFGS, not every %. For "
            "example at T = 2, 10<sup>-6</sup> was tried at 0.25% and 2% only.",
            "The two follow-up runs (T = 2.5 to 250,000 steps, T = 3 to 200,000) were stopped at 180,000 and 140,000 "
            "steps, after the cloud machine restarted twice under them and both had flattened out (Figure 21.2). "
            "They resumed from saved checkpoints; L-BFGS restarts its memory there, a small change from one "
            "unbroken run. Reaching the last four cells would need a different method (for example time-marching, "
            "or a larger network), not more data or more steps.",
            "Long L-BFGS is slow on CPU: 60,000 steps at T = 3 took 5.6 hours on one core."])
    s += [P("21.7 Files", h2)]
    s += [P("Code: <font face='Courier'>scripts/pct_sweep.py</font> (runs, with --lbfgs, --maxfun, --seed and "
            "--ckpt-dir), <font face='Courier'>scripts/pct_table.py</font> (the table) and "
            "<font face='Courier'>scripts/pct_report_section.py</font> (this section). Results in "
            "results/pct_sweep/: pct_needed_table_filled.png and .xlsx (with Reasons and All runs sheets), "
            "pct_needed_filled.csv, pct_sweep_all_runs.csv, and extra/ (one CSV per batch of runs, reasons.csv, "
            "and checkpoints/ with the saved weights and error-per-checkpoint files).", body)]
    return s
