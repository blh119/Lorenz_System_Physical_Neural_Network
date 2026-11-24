"""Report Sections 22 to 25: results that were saved but not yet in the report.

22. The data-needed table, counted by order of magnitude
23. Inside the warm-start hand-off: what happens to the weights
24. The Lorenz path on [0, 20] and its wing flips
25. Wing flips and where the data must go ([0, 2] and [13, 15])

Builds two new graphs, the section pages (reportlab) and a new contents page, then
writes a new Lorenz_PINN_Report.pdf: old page 1, new contents, old pages 3+, new
sections. The previous report is kept as Lorenz_PINN_Report_before_section22.pdf.
Same page style as pct_report_section.py (Section 21).

    python report_sections_22_25.py /mnt/project-files/results
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle, PageBreak)

RES = sys.argv[1] if len(sys.argv) > 1 else "../results"
REPORT = os.path.join(RES, "Lorenz_PINN_Report.pdf")
BACKUP = os.path.join(RES, "Lorenz_PINN_Report_before_section22.pdf")
WORK = os.environ.get("WORK", "/tmp")
TITLE = "Teaching a Neural Network the Lorenz System"
BLUE = colors.HexColor("#1f5fa8")
NEW_TITLES = ["The data-needed table, counted by order of magnitude",
              "Inside the warm-start hand-off: what happens to the weights",
              "The Lorenz path on [0, 20] and its wing flips",
              "Wing flips and where the data must go"]
FLIPS = [0.513, 13.549, 14.335, 17.243, 18.072, 18.850]

body = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=15, spaceAfter=6)
h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17, leading=22, spaceAfter=10)
h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13, leading=17, spaceBefore=10, spaceAfter=6)
cap = ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=8.5, leading=11, alignment=TA_CENTER,
                     textColor=colors.HexColor("#555555"), spaceAfter=10)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=2, spaceAfter=4)
cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.5, leading=10.5)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
boxh = ParagraphStyle("boxh", fontName="Helvetica-Bold", fontSize=9.5, textColor=BLUE, spaceAfter=4)
P = Paragraph
C = "<font face='Courier'>%s</font>"


def sci(x, d=1):
    m, e = f"{x:.{d}e}".split("e")
    return f"{m} × 10<super>{int(e)}</super>"


def e(n):
    return f"10<super>{n}</super>"


def box(title, text, bg="#eaf1fb"):
    paras = [Paragraph(title, boxh)] + [Paragraph(t, body) for t in (text if isinstance(text, list) else [text])]
    t = Table([[paras]], colWidths=[6.5 * inch])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, -1), (-1, -1), 8)]))
    return t


def takeaway(text):
    return box("Key takeaway", text, "#fdf0e6")


def grid(rows, widths, header=True, bold_cells=()):
    data = [[Paragraph(str(c), cellb if (header and i == 0) or (i, j) in bold_cells else cell)
             for j, c in enumerate(r)] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")))
    for (i, j) in bold_cells:
        st.append(("BACKGROUND", (j, i), (j, i), colors.HexColor("#e3f3e3")))
    t.setStyle(TableStyle(st))
    return t


def B(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def img(path, width=6.5, max_h=7.6):
    w, h = PILImage.open(path).size
    wi, hi = width, width * h / w
    if hi > max_h:
        wi, hi = max_h * w / h, max_h
    return Image(path, width=wi * inch, height=hi * inch)


def fig(path, caption, width=6.5, max_h=7.6):
    return KeepTogether([img(path, width, max_h), P(caption, cap)])


# ---------------------------------------------------------------- layouts (same rules as location_sweep.py)
LAYOUTS = ["even", "random", "early", "middle", "late", "first_half", "second_half"]
LABELS = {"even": "Spread evenly", "random": "Random", "early": "First third", "middle": "Middle third",
          "late": "Last third", "first_half": "First half", "second_half": "Second half"}
SEGMENTS = {"even": (0, 1), "early": (0, 1 / 3), "middle": (1 / 3, 2 / 3), "late": (2 / 3, 1),
            "first_half": (0, .5), "second_half": (.5, 1)}


def layout_times(layout, n, T, seed=99):
    if layout == "random":
        return np.sort(np.random.default_rng(seed).random(n)) * T
    a, b = SEGMENTS[layout]
    a, b = a * T, b * T
    return a + (np.arange(n) + 0.5) * (b - a) / n


# ---------------------------------------------------------------- new graphs
def plot_first_flip(loc):
    """[0, 2]: where each layout's points sit relative to the first flip, and whether the run worked."""
    ref = pd.read_csv(os.path.join(RES, "reference", "x_reference_T0-20.csv"))
    ref = ref[ref["t"] <= 2.0001]
    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax2 = ax.twinx()
    ax2.plot(ref["t"], ref.iloc[:, 1], color="#bbbbbb", lw=1.2, zorder=0)
    ax2.set_ylabel("reference x(t)", color="#888888")
    ax2.axhline(0, color="#dddddd", lw=0.6)
    rows = []
    for lay in LAYOUTS:
        for n, pct in [(20, 0.1), (50, 0.25)]:
            rows.append((lay, n, pct))
    for k, (lay, n, pct) in enumerate(rows):
        yk = len(rows) - 1 - k
        r = loc[(loc["T"] == 2) & (loc["layout"] == lay) & (loc["pct_reference"] == pct)].iloc[0]
        ok = r.rel_l2 < 1e-4
        c = "#2a9d4b" if ok else "#d1495b"
        t = layout_times(lay, n, 2.0)
        ax.vlines(t, yk - 0.32, yk + 0.32, color=c, lw=1.1)
        ax.text(2.03, yk, f"{r.rel_l2:.1e}", va="center", fontsize=8.5, color=c,
                fontweight="bold" if ok else "normal")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{LABELS[l].lower()}, {n} pts" for l, n, _ in rows][::-1], fontsize=8.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.axvline(FLIPS[0], color="#c00000", ls="--", lw=1.2)
    ax.text(FLIPS[0] + 0.015, len(rows) - 0.45, "first wing flip\nt ≈ 0.513", color="#c00000", fontsize=8.5,
            va="top")
    ax.axvspan(0.2, FLIPS[0], color="#fff2cc", zorder=-1)
    ax.text(0.356, -0.55, "big swing", ha="center", fontsize=8, color="#8a6d00")
    ax.set_xlim(0, 2.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    ax.set_xlabel("t   (numbers on the right: relative L2 over [0, 2]; green = under 1e-4)")
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_title("One network on [0, 2]: where each layout's data points sit (ticks), green = worked, red = failed",
                 fontsize=10.5)
    path = os.path.join(WORK, "first_flip_T2.png")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    return path


def plot_t13_preds():
    """[13, 15]: what the networks actually drew, for three layouts at 0.25%."""
    d = os.path.join(RES, "location_sweep", "runs_t13-15")
    picks = [("even", "spread evenly (worked)", "#2a9d4b"),
             ("first_half", "first half (sees flip 1)", "#d1495b"),
             ("second_half", "second half (sees flip 2)", "#7b4fa0")]
    fig, axs = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True)
    ref = pd.read_csv(os.path.join(d, "pred_t13-15_pct0.25_even.csv"))
    for a, v in zip(axs, "xyz"):
        a.plot(ref["t"], ref[f"{v}_pos"], color="black", lw=2.2, label="RK4 reference")
        for lay, lab, c in picks:
            p = pd.read_csv(os.path.join(d, f"pred_t13-15_pct0.25_{lay}.csv"))
            a.plot(p["t"], p[f"{v}_pinn"], color=c, lw=1.3, ls="--" if lay == "even" else "-", label=lab)
        for f in FLIPS[1:3]:
            a.axvline(f, color="#c00000", ls=":", lw=1)
        a.set_ylabel(f"{v}(t)"); a.grid(alpha=0.3)
    axs[0].legend(fontsize=8, ncol=4, loc="lower left")
    axs[-1].set_xlabel("t   (red dotted lines: wing flips at 13.549 and 14.335)")
    axs[0].set_title("One network on [13, 15], 50 data points (0.25%): the prediction for three layouts", fontsize=10.5)
    path = os.path.join(WORK, "location_t13-15_predictions.png")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    return path


# ---------------------------------------------------------------- sections
def section22():
    om = pd.read_csv(os.path.join(RES, "pct_sweep", "pct_needed_filled_order_of_magnitude.csv"))
    s = [P("22. " + NEW_TITLES[0], h1)]
    s += [box("In plain words",
              ["Section 21 filled the data-needed table strictly: a cell counts only if the error is at or under "
               "the level. Four cells stayed empty. Brian asked for a second version where a result counts if it is "
               f"the same order of magnitude as the level. For example, 3.9 × {e(-6)} counts as reaching "
               f"{e(-6)}, because it is \"a few times {e(-6)}\".",
               "<b>Short answer:</b> counted this way, 28 of the 30 cells are filled. Only T = 3 at "
               f"{e(-5)} and {e(-6)} still fail, because the best one network ever did on [0, 3] is "
               f"5.8 × {e(-4)}."])]
    s += [Spacer(1, 8), P("22.1 The rule", h2)]
    s += [P(f"A cell for level {e('-k')} is filled when the run's relative L2 over [0, T] is under "
            f"10 × {e('-k')}, i.e. it starts with \"× {e('-k')}\" when written in scientific notation. Every "
            "run is one from Sections 18 or 21; nothing was retrained. The cell shows the smallest % of the RK4 "
            "reference points that did it.", body)]
    levels = ["≤ 1e-6", "≤ 1e-5", "≤ 1e-4", "≤ 1e-3", "≤ 1e-2", "≤ 1e-1"]
    rows = [["T", f"{e(-6)}", f"{e(-5)}", f"{e(-4)}", f"{e(-3)}", f"{e(-2)}", f"{e(-1)}"]]
    for T in [1, 1.5, 2, 2.5, 3]:
        row = [f"{T:g}"]
        for lev in levels:
            r = om[(om["T"] == T) & (om["level"] == lev)].iloc[0]
            if r.status == "failed":
                row.append(f"failed (best {sci(r.rel_l2)})")
            else:
                star = "*" if isinstance(r.setting, str) and r.setting else ""
                row.append(f"{r.pct:g}%{star}")
        rows.append(row)
    s += [grid(rows, [0.45 * inch] + [1.0 * inch] * 6)]
    s += [P("Smallest % of the reference points that reached each level, counted by order of magnitude. "
            "* needed longer L-BFGS training (20,000 to 100,000 steps instead of 5,000), as in Section 21.", cap)]
    s += [fig(os.path.join(RES, "pct_sweep", "pct_needed_table_filled_order_of_magnitude.png"),
              "Figure 22.1. The same table as an image (also saved as an Excel file).", 6.3)]
    s += [P("22.2 What changes from the strict table", h2)]
    s += B([f"<b>T = 2.5 at {e(-6)}:</b> the best run reached 3.9 × {e(-6)} with 1% of the points (250) after "
            "long L-BFGS, so it now counts.",
            f"<b>T = 3 at {e(-4)}:</b> the best run reached 5.8 × {e(-4)} (5.2 × {e(-4)} at its lowest) with 10% of "
            "the points (3,000), so it now counts.",
            f"<b>T = 3 at {e(-5)} and {e(-6)}</b> still fail. Every amount from 10% to 100% of the points and "
            "three seeds got stuck, so more data would not fill them.",
            f"Some cells show a smaller % than the strict table. At T = 2 and {e(-4)}, 0.1% gave 1.5 × {e(-4)}, "
            "which is the right order of magnitude though not strictly under the level."])
    s += [takeaway(f"Counted by order of magnitude, one network reaches \"{e(-6)}\" out to T = 2.5 with at most "
                   "1% of the data, and reaches T = 3 only at the 10<super>-4</super> level, with 10% of the data "
                   "and long training. The limit at T = 3 is the training, not the data.")]
    s += [P("22.3 Files", h2)]
    s += [P("results/pct_sweep/: pct_needed_table_filled_order_of_magnitude.png and .xlsx, "
            "pct_needed_filled_order_of_magnitude.csv (each cell with its run, error and, for failed cells, "
            "the reason).", body)]
    return s


def section23():
    rms = pd.read_csv(os.path.join(RES, "handoff_weights", "simple", "weight_rms.csv"))
    hw = os.path.join(RES, "handoff_weights")
    s = [PageBreak(), P("23. " + NEW_TITLES[1], h1)]
    s += [box("In plain words",
              ["Section 20 showed that the warm start (copying the previous window's weights into the next "
               "window's network) makes time-marching worse and makes long runs break. Brian asked what is "
               "actually going on inside the network at each hand-off. So two runs were saved weight by weight, "
               "window by window, and compared: one with the warm start and one with fresh random weights.",
               "<b>Short answer:</b> with the warm start the weights get bigger at every hand-off and never shrink. "
               "Bigger weights push the network's tanh units into their flat ends, where they stop learning. By "
               "t = 4 most units were stuck, and the window starting at t = 4.25 could not be trained at all. "
               "Fresh weights start small every time and stay healthy. The hand-off error gives no warning before "
               "the failure; the weight size does."])]
    s += [Spacer(1, 8), P("23.1 Setup", h2)]
    s += [P("Two time-marching runs, identical except for the warm start: T = 5 in 20 windows of 0.25 time units "
            "(hand-offs at t = 0.25, 0.5, ..., 5), data everywhere, seed 99, 10,000 Adam epochs and up to 5,000 "
            "L-BFGS steps per window, hard initial condition. Both runs share window 1 exactly, so every "
            "difference after that comes from the hand-off. Every window's trained weights were saved "
            f"({C % 'time_marching.py --save-dir'}) and analysed afterwards in NumPy, so nothing was retrained. "
            "Over [0, 5] the warm run ended with a relative L2 of 1.7 × 10<super>-2</super> (it broke at t = 4.25); "
            "the fresh run ended at 2.0 × 10<super>-7</super>.", body)]

    s += [P("23.2 The one number to watch: weight size at each hand-off", h2)]
    s += [P("At each hand-off, take all 12,736 weights and biases of the four hidden layers (64 units each) of the "
            "network being handed over, compute their RMS (root mean square), and divide by the RMS of the random "
            "network the run started from (0.123). So 1.0 means \"the size of a brand new network\".", body)]
    s += [fig(os.path.join(hw, "simple", "weight_rms_both.png"),
              "Figure 23.1. Hidden-weight size at every hand-off (top) and the final training loss of the window "
              "being handed over (bottom). Pink: warm start. Green: fresh weights. Red X's: windows after the warm "
              "run went off the path; circle: its last good hand-off (t = 4.25).", 6.0, 5.0)]
    w = rms[rms["run"] == "warm"].set_index("handoff_t")
    f = rms[rms["run"] == "fresh"].set_index("handoff_t")
    rows = [["Hand-off at t", "Warm: weight size", "Fresh: weight size", "Warm: hand-off error", "Fresh: hand-off error"]]
    for t in [1, 2, 3, 4, 4.25, 4.5, 5]:
        rows.append([f"{t:g}", f"{w.loc[t, 'rms_handoff_norm']:.2f}", f"{f.loc[t, 'rms_handoff_norm']:.2f}",
                     sci(w.loc[t, "handoff_err_3d"]), sci(f.loc[t, "handoff_err_3d"])])
    s += [grid(rows, [1.0 * inch, 1.3 * inch, 1.3 * inch, 1.45 * inch, 1.45 * inch])]
    s += [P("Weight size: hidden-weight RMS divided by that of a new random network. Hand-off error: 3D distance "
            "from the true state at the end of the window. Separate warm and fresh versions of Figure 23.1 are in "
            "results/handoff_weights/simple/.", cap)]
    s += B(["<b>Warm start:</b> the size climbs at every hand-off and never comes back down: 2.09 at t = 1, 2.81 "
            "at t = 2, 3.33 at t = 3, 4.13 at t = 4 and 4.35 at t = 4.25. The next window (4.25 to 4.5) went off "
            "the path, ending 2.8 away from the reference, and the size jumped to 6.45 by t = 5.",
            "<b>Fresh weights:</b> every window starts again at 1.0 and ends at about 1.2 to 1.3. The one spike "
            "(2.30 at t = 0.75) is a hard stretch of the path, and the next fresh network reset it.",
            f"<b>The loss and the hand-off error give no warning.</b> Both runs end every good window between "
            f"{e(-10)} and {e(-8)} in loss and hand off errors of {e(-7)} to {e(-5)}, warm included, right up to "
            "t = 4.25. Then the warm loss jumps to 2.4 in one window. The weight size climbed steadily the whole "
            "time, so it is the early warning."])

    s += [P("23.3 The detailed graphs", h2)]
    s += [P("The six graphs below break the hand-off down further. Each is explained in "
            "results/handoff_weights/graph_guide.md.", body)]
    s += [fig(os.path.join(hw, "handoff_errors.png"),
              "Figure 23.2. Top: 3D error along the path. Bottom: the error each window hands to the next. Both runs "
              f"hand over {e(-7)} to {e(-5)} up to t = 4.25; the warm run then jumps to 2.8 in one window.", 5.8, 4.2)]
    s += [fig(os.path.join(hw, "handoff_losses.png"),
              "Figure 23.3. Loss of each window at epoch 1, after 10,000 Adam epochs, and after L-BFGS.", 6.5)]
    s += B(["<b>Epoch 1:</b> the copied network is a worse starting guess than random weights, about 3× worse in "
            "every window (10 to 17 against 3 to 6).",
            f"<b>After Adam:</b> fresh stays at {e(-7)} to {e(-6)} in every window. Warm creeps up from {e(-6)} to "
            f"1.5 × {e(-4)} by t = 4, about 100× worse. This is the clearest warning in the losses.",
            f"<b>After L-BFGS:</b> both reach about {e(-9)}, so L-BFGS hides the problem until the window at "
            "t = 4.25, which stays at 2.4."])
    s += [fig(os.path.join(hw, "starting_guess.png"),
              "Figure 23.4. What each network draws before any training, at three hand-offs. Black: reference. "
              "Pink: the copied warm network. Green: a fresh network.", 6.0, 3.9)]
    s += [P("The copied network's starting guess is the previous window's curve redrawn on the new window, full "
            "size and often going the wrong way (in window 10, y should rise and the guess falls). Every window "
            "feeds the network the same local time on [-1, 1], so nothing tells it that the new window comes after "
            "the old one. The fresh guess is nearly flat: wrong too, but less wrong. That is why warm windows start "
            "with a 3× higher loss.", body)]
    s += [fig(os.path.join(hw, "weight_size.png"),
              "Figure 23.5. Size (norm) of each layer's weights, and of all 12,803 weights, at the end of every window.",
              6.5)]
    s += [P("Warm weights grow in every window: the total goes from 16.5 to 61.5 by t = 4, then to 100 after the "
            "failure, and hidden layers 2 to 4 grow 3 to 4 times. Fresh weights end every window at about 16 to 18 "
            "(one hard window at t = 0.5 reached 32, and the next fresh window reset it). Nothing in the training "
            "shrinks the weights, so under the warm start each window's training is stacked on top of all the "
            "earlier ones.", body)]
    s += [fig(os.path.join(hw, "unit_health.png"),
              "Figure 23.6. Top: share of hidden units stuck in the flat ends of tanh (input bigger than 2, slope "
              "under 0.07). Bottom: how many different shapes the 64 units of a layer make (rank of the layer's "
              "outputs).", 6.5)]
    s += B(["<b>Layer 2:</b> warm goes from 0% stuck to 88% by t = 4; fresh stays under 1% (apart from the hard "
            "window at t = 0.5).",
            "<b>Layers 3 and 4:</b> warm sits at 85% to 97% stuck from t = 0.5 on; fresh moves between 5% and 77%.",
            "<b>The failed window:</b> at t = 4.25 layers 3 and 4 were 100% stuck and made just one shape. Every unit "
            "was pinned at +1 or -1, so the network could only output a near-constant: the \"almost straight line\" "
            "seen in failed windows in Section 17."])
    s += [fig(os.path.join(hw, "weight_movement.png"),
              "Figure 23.7. Left: how far the weights travel in each window. Middle: how much of that trip is done "
              "after 1, 10, 100, 1,000 and 2,000 Adam steps. Right: the length of Adam's first step, as a share of a "
              "full step.", 6.5)]
    s += B(["<b>No head start:</b> warm windows travel at least as far as fresh ones (about 10 per window for fresh, "
            "10 to 20 for warm). The copied weights were not closer to the answer, which is why the warm start saved "
            "no time in Section 20.",
            "<b>Most movement is early:</b> both runs do most of their moving in the first 100 Adam steps, so "
            "whatever the copied weights knew is mostly overwritten right away.",
            "<b>Fewer weights still learn:</b> Adam's first step moves every weight that has any gradient by about "
            "the learning rate. Fresh takes a full first step every window; warm's shrinks from 1.0 to 0.5 at t = 4 "
            "and 0.14 after the failure. That points to about three quarters of the warm weights getting almost no "
            "gradient by t = 4 (a rough reading), the same saturation as Figure 23.6."])
    s += [takeaway("The warm start does not fail because of the error it passes on. It fails because the copied "
                   "weights keep growing, the growth saturates the tanh units, and the network slowly loses the units "
                   "it can train, until one window cannot be trained at all. Fresh weights avoid that by starting "
                   "small every time. If a warm start is still wanted, cheap fixes to try are shrinking the copied "
                   "weights toward zero before each window, copying only some layers, or a small penalty on weight "
                   "size; the weight-size graph would show at once whether a fix stops the build-up.")]
    s += [P("23.4 Limits", h2)]
    s += B(["One seed (99) and one window length (0.25). The pattern matches Section 20's three seeds at T = 5 and "
            "the T = 20 runs, but the weight graphs themselves are one run each.",
            "The \"share of weights still learning\" from Adam's first step is an estimate, not a direct count."])
    s += [P("23.5 Files", h2)]
    s += [P(f"Code: {C % 'scripts/handoff_analysis.py'} (the six detailed graphs and handoff_summary.csv) and "
            f"{C % 'scripts/handoff_weight_rms.py'} (the weight-size graphs and weight_rms.csv). Results in "
            "results/handoff_weights/: the six graphs, graph_guide.md, handoff_summary.csv (one row per window and "
            "run), runs/ (saved weights and losses of every window), and simple/ (weight_rms_both, _warm and "
            "_fresh graphs, weight_rms.csv and a README).", body)]
    return s


def section24():
    d3 = os.path.join(RES, "3d")
    s = [PageBreak(), P("24. " + NEW_TITLES[2], h1)]
    s += [box("In plain words",
              ["The Lorenz path circles two \"wings\", one around each of the two fixed points "
               "C<sub>±</sub> = (±8.49, ±8.49, 27). A <b>wing flip</b> is the moment the path crosses from one wing "
               "to the other, which is when x changes sign. Brian asked when the flips happen on [0, 20], the time "
               "range of the long time-marching runs in Sections 17 and 20, and for pictures of the path.",
               "<b>Short answer:</b> there are 6 flips on [0, 20]. After the first one at t ≈ 0.51 the path stays "
               "on the left wing for about 13 time units, then flips 5 times between t ≈ 13.5 and 18.9. Those late "
               "flips line up with the hardest stretch for the PINN windows in Section 17 (around t = 12 to 13 and "
               "after)."])]
    s += [Spacer(1, 8), P("24.1 The flip times", h2)]
    sides = ["left (x < 0)", "right (x > 0)", "left", "right", "left", "right"]
    rows = [["Flip", "t", "Moves to wing", "Time since previous flip"]]
    prev = 0.0
    for i, (t, sd) in enumerate(zip(FLIPS, sides), 1):
        rows.append([str(i), f"{t:.3f}", sd, f"{t - prev:.2f}" if i > 1 else "(from t = 0) 0.51"])
        prev = t
    s += [grid(rows, [0.6 * inch, 1.0 * inch, 1.6 * inch, 2.2 * inch])]
    s += [P("RK4 reference from (1, 1, 1), dt = 10<super>-4</super>, σ = 10, ρ = 28, β = 8/3. A flip is a sign "
            "change of x between two reference points.", cap)]
    s += [P("<b>What leads up to a flip.</b> Each loop around a wing's centre spirals a little further out than the "
            "one before. On the left wing the x swings grow from about -7 to -9 near t = 2 to about -2 to -15 by "
            "t = 13, and the dips in z get deeper (from about 24 down to about 8). Once a loop swings far enough "
            "out, it passes close to the z-axis, where x is near 0, and from there it is thrown across to the other "
            "wing, with a big spike in z (41 at t ≈ 14.3). Which side it lands on depends on exactly how close it "
            "passes to the z-axis, the most sensitive part of the attractor, where small errors grow fastest.",
            body)]
    s += [fig(os.path.join(d3, "lorenz_3d_static.png"),
              "Figure 24.1. The RK4 reference path in 3D for t in [0, 20], coloured by time (purple at the start, "
              "yellow at the end). Orange dots: the 6 wing flips. Red X's: the fixed points C±. White dot: the start "
              "(1, 1, 1).", 5.6, 5.2)]
    s += [fig(os.path.join(d3, "lorenz_xyz_vs_time.png"),
              "Figure 24.2. x(t), y(t) and z(t) on [0, 20] with a dashed line at each wing flip; the flip times are "
              "written along the top.", 6.5)]
    s += [P("24.2 What it shows for the PINN", h2)]
    s += B(["<b>Flips are where the path is hardest to follow.</b> In Section 17 the long runs struggled most around "
            "t ≈ 12 to 13 and after; 5 of the 6 flips fall between 13.5 and 18.9. The data-only-on-[0, 1] run at "
            "T = 20 broke around t = 12, just before the flips start.",
            "<b>The first flip is inside every short window.</b> The flip at t ≈ 0.51 (right after the big swing "
            "near t = 0.35) falls inside [0, 1] and [0, 2], the windows used for the one-network tests in Sections "
            "15 to 21. Section 25 tests how much that matters."])
    s += [P("24.3 Moving and interactive versions", h2)]
    s += [P("These are in results/3d/ and can't be shown on paper: lorenz_3d_motion.mp4 (a 20-second video, fixed "
            "camera, where a red dot traces the path and the title counts the flips so far), lorenz_3d_motion.gif "
            "(the same as a GIF, for slides that can't play video) and lorenz_3d_interactive.html (drag to rotate; "
            "hovering shows t, x, y and z; works offline). The full reference x(t) on [0, 20] at dt = 10<super>-4</super> "
            "is in results/reference/x_reference_T0-20.csv.", body)]
    s += [P("24.4 Files", h2)]
    s += [P(f"Code: {C % 'scripts/make_3d_charts.py'} (all five outputs; {C % '--csv'} and {C % '--out'} choose the "
            "input and output folder). Results in results/3d/.", body)]
    return s


def section25(loc, loc13, figs):
    s = [PageBreak(), P("25. " + NEW_TITLES[3], h1)]
    s += [box("In plain words",
              ["Section 19 found that one network must have data at the start of its window. Looking at the [0, 2] "
               "results again with the flip times from Section 24, Brian noticed that the layouts that worked were "
               "the ones that saw the first wing flip (t ≈ 0.51). That raised a question: <b>if the network sees one "
               "flip, can it predict the next one?</b> To test it, the exact Section 19 experiment was repeated on "
               "[13, 15], a 2-unit window with two flips in it (t ≈ 13.549 and 14.335).",
               "<b>Short answer: no.</b> On [0, 2], seeing the first flip was necessary but not enough on its own. On "
               "[13, 15], seeing one flip did not let the network predict the next: only data spread across the "
               f"whole window (spread evenly or random) reached {e(-4)}. The layouts that saw only the first flip "
               "did not even reproduce that flip; the network fell onto a different exact solution of the equations, "
               "with x and y stuck at 0."])]
    s += [Spacer(1, 8), P("25.1 Back to [0, 2]: the first flip", h2)]
    s += [P("Same runs as Section 19.3 (one network on [0, 2], seed 99, 20 or 50 points, i.e. 0.1% or 0.25% of the "
            "20,001 reference points), now counted against the first flip at t ≈ 0.513.", body)]
    rows = [["Placement", "20 pts: before flip", "20 pts: rel. L2", "50 pts: before flip", "50 pts: rel. L2"]]
    bold = []
    for i, lay in enumerate(LAYOUTS, 1):
        row = [LABELS[lay]]
        for j, (n, pct) in enumerate([(20, 0.1), (50, 0.25)]):
            r = loc[(loc["T"] == 2) & (loc["layout"] == lay) & (loc["pct_reference"] == pct)].iloc[0]
            nb = int((layout_times(lay, n, 2.0) < FLIPS[0]).sum())
            row += [str(nb), sci(r.rel_l2)]
            if r.rel_l2 < 1e-4:
                bold.append((i, 2 + 2 * j))
        rows.append(row)
    s += [grid(rows, [1.3 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch], bold_cells=bold)]
    s += [P(f"\"Before flip\": number of data points at t &lt; 0.513. Green: under {e(-4)}.", cap)]
    s += [fig(figs["first_flip"],
              "Figure 25.1. One network on [0, 2]. Each row is one layout and amount; the ticks are its data times "
              "(green = the run reached 1e-4, red = it failed). Grey: reference x(t). Dashed red: the first wing "
              "flip. Shaded: the big swing before it.", 6.5, 4.3)]
    s += B(["<b>Every layout with no data before the flip failed</b> (middle third, last third, second half, at both "
            "amounts: 0.6 to 0.8 error).",
            "<b>Seeing the flip was not enough on its own.</b> Random (2 or 8 points before the flip), first third "
            "with 20 points (15 before the flip) and spread evenly with 50 points (13 before the flip) all saw it and "
            "still failed. The layouts that worked had points packed through the swing and the flip, with no long "
            "gap there; Section 19 noted that single runs at T = 2 can flip between working and failing.",
            "So on [0, 2] the data must cover the first flip, and cover it densely, but the count of points alone "
            "doesn't decide it."])

    s += [P("25.2 The [13, 15] test: does seeing one flip predict the next?", h2)]
    s += [P("Everything is the same as Section 19 and 25.1, moved to [13, 15]: one 4 × 64 tanh network, float64, "
            "Adam for 10,000 epochs then L-BFGS capped at 5,000 steps, seed 99, 20 or 50 points (0.1% or 0.25% of "
            "the 20,001 reference points), the same seven layout rules and the same random draw, shifted by 13. "
            "The network starts from the exact RK4 state at t = 13, (-5.737, -9.833, 13.897), built in as a hard "
            "initial condition. The sampling code reproduces all 28 earlier T = 1 and T = 2 layouts exactly; as a "
            f"check, the [0, 2] first-half 50-point run was repeated and gave 7.3 × {e(-5)} (2.6 × {e(-5)} in "
            f"Section 19, both under {e(-4)}). \"First third\" and \"first half\" see flip 1 only; \"last third\" "
            "sees flip 2 only.", body)]
    rows = [["Placement", "Sees", "0.1% (20 pts)", "0.25% (50 pts)"]]
    sees = {"even": "both flips", "random": "both flips", "early": "flip 1", "middle": "neither (ends just before flip 2)",
            "late": "only after flip 2", "first_half": "flip 1", "second_half": "flip 2"}
    bold = []
    for i, lay in enumerate(LAYOUTS, 1):
        row = [LABELS[lay], sees[lay]]
        for j, pct in enumerate([0.1, 0.25]):
            r = loc13[(loc13["layout"] == lay) & (loc13["pct_reference"] == pct)].iloc[0]
            row.append(sci(r.rel_l2))
            if r.rel_l2 < 1e-4:
                bold.append((i, 2 + j))
        rows.append(row)
    s += [grid(rows, [1.4 * inch, 1.8 * inch, 1.5 * inch, 1.5 * inch], bold_cells=bold)]
    s += [P(f"Relative L2 over all of [13, 15]; green = under {e(-4)}.", cap)]
    s += [fig(os.path.join(RES, "location_sweep", "location_t13-15.png"),
              "Figure 25.2. One network on [13, 15]. Left: error for each layout and amount. Right: where the 0.25% "
              "points sit, over the reference x(t), with both flips marked.", 6.5)]
    s += [fig(figs["t13"],
              "Figure 25.3. What three of the 50-point networks actually drew on [13, 15]. Black: reference. Green "
              "dashed: spread evenly, on top of the reference. Red: first half, which falls onto x = y = 0 just before "
              "flip 1. Purple: second half, which misses the start and only rejoins the path after flip 2.", 6.5, 5.0)]
    s += B([f"<b>Only data across the whole window worked.</b> Spread evenly ({sci(loc13.loc[loc13.name == 't13-15_pct0.25_even', 'rel_l2'].iloc[0])} "
            f"at 50 points) and random ({sci(loc13.loc[loc13.name == 't13-15_pct0.1_random', 'rel_l2'].iloc[0])} at 20 "
            "points) both placed the two flips at the right times. Every partial layout failed, at 0.4 to 0.8 error, "
            "with nearly the same error at 20 and 50 points, so this is a consistent trap, not bad luck.",
            "<b>Seeing flip 1 did not predict flip 2, or even flip 1.</b> First half and first third have data on "
            "both sides of flip 1, yet their error is already 10<super>-2</super> by t ≈ 13.48. Near the flip the true "
            "path passes within 0.56 of the z-axis, where x = y = 0. x = y = 0 with z decaying (z' = -βz) is also an "
            "exact solution of the Lorenz equations, so the physics loss is happy there, and the network settles on it and "
            "ignores the data that comes after.",
            "<b>Seeing only flip 2 doesn't recover the start.</b> Second half and middle third miss the first stretch, "
            "draw a wrong path through flip 1 and only rejoin the reference after flip 2. Last third never gets back.",
            "<b>Why [0, 2] and [13, 15] differ.</b> On [0, 2] the first half worked: that window has one big swing "
            "and its flip stays at least 8.7 from the z-axis. On [13, 15] flip 1 passes within 0.56 of it, right "
            "next to a wrong but valid solution."])
    s += [takeaway("Data must cover each wing flip. Seeing one flip does not let one network predict the next one, "
                   "because near a flip the path can pass close to the z-axis, where the network can fall onto the "
                   "x = y = 0 solution that satisfies the physics. For windows with several flips, spread the data "
                   "over the whole window, or use short windows with data in each (time-marching, Sections 16, 17 "
                   "and 20).")]
    s += [P("25.3 Limits", h2)]
    s += B(["One run per layout and amount, seed 99, as in Section 19. The [13, 15] failures were the same at both "
            "amounts, which makes them look systematic, but other seeds were not tried.",
            "One shifted window. Other windows with flips (for example [17, 19], which has three) were not tested.",
            "L-BFGS capped at 5,000 steps, as in Sections 16, 18 and 19."])
    s += [P("25.4 Files", h2)]
    s += [P(f"Code: {C % 'scripts/location_sweep.py'} (the layout rules and the runs on any window: "
            f"{C % '--t0 13 --T 2'}). Results in results/location_sweep/: location_results_t13-15.csv (every run, "
            "with its number of points, first and last data time and largest gap), location_t13-15.png, and "
            "runs_t13-15/ (the prediction of every run). The [0, 2] runs are in location_results.csv (Section 19).",
            body)]
    return s


# ---------------------------------------------------------------- pages
def footer_factory(first_page):
    def footer(canv, doc):
        canv.saveState()
        canv.setFont("Helvetica", 8); canv.setFillColor(colors.HexColor("#555555"))
        canv.drawString(0.9 * inch, 0.55 * inch, TITLE)
        canv.drawRightString(7.6 * inch, 0.55 * inch, f"Page {first_page + doc.page - 1}")
        canv.restoreState()
    return footer


def build_doc(path, story, first_page):
    doc = BaseDocTemplate(path, pagesize=letter, leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                          topMargin=0.8 * inch, bottomMargin=0.9 * inch)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=footer_factory(first_page))])
    doc.build(story)


def contents_page(path, titles):
    st = ParagraphStyle("c", fontName="Helvetica", fontSize=10, leading=16)
    story = [Paragraph("Contents", ParagraphStyle("ch", fontName="Helvetica-Bold", fontSize=20,
                                                  textColor=BLUE, spaceAfter=14))]
    story += [Paragraph(f"{i}. {t}", st) for i, t in enumerate(titles, 1)]
    build_doc(path, story, 2)


def main():
    loc = pd.read_csv(os.path.join(RES, "location_sweep", "location_results.csv"))
    loc13 = pd.read_csv(os.path.join(RES, "location_sweep", "location_results_t13-15.csv"))
    figs = dict(first_flip=plot_first_flip(loc), t13=plot_t13_preds())

    if not os.path.exists(BACKUP):
        shutil.copy(REPORT, BACKUP)
    old = PdfReader(BACKUP)
    first = len(old.pages) + 1
    sec = os.path.join(WORK, "sections22_25.pdf")
    build_doc(sec, section22() + section23() + section24() + section25(loc, loc13, figs), first)

    import re
    lines = old.pages[1].extract_text().replace("Contents", "\n").splitlines()
    titles = [m.group(1) for m in (re.match(r"\s*\d+\.\s+(.*)", l) for l in lines) if m]
    titles = [t for t in titles if t not in NEW_TITLES] + NEW_TITLES
    toc = os.path.join(WORK, "contents.pdf")
    contents_page(toc, titles)

    w = PdfWriter()
    w.add_page(old.pages[0])
    w.add_page(PdfReader(toc).pages[0])
    for p in old.pages[2:]:
        w.add_page(p)
    for p in PdfReader(sec).pages:
        w.add_page(p)
    w.add_metadata({"/Title": TITLE})
    tmp = REPORT + ".tmp"
    with open(tmp, "wb") as f:
        w.write(f)
    os.replace(tmp, REPORT)
    for k, v in figs.items():
        print("figure", k, v)
    print(f"report: {len(w.pages)} pages, section 22 starts on page {first}")


if __name__ == "__main__":
    main()
