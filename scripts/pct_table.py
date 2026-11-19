"""Build the 'smallest % of data that reaches each error level' table (PNG, Excel, CSV).

Reads the original sweep (pct_sweep_results.csv) plus the fill-in runs in extra/,
and for each T and level picks the smallest % that got there. Runs at the
original settings (5,000 L-BFGS steps, seed 99) are preferred; a cell only
reached with longer L-BFGS or another seed says so. Cells no run reached are
marked FAILED with the reason given in reasons.csv (T, level, reason).

    python pct_table.py ../results/pct_sweep
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEVELS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
LEVEL_TXT = ["≤ 1e-6", "≤ 1e-5", "≤ 1e-4", "≤ 1e-3", "≤ 1e-2", "≤ 1e-1"]


def load(d):
    base = pd.read_csv(os.path.join(d, "pct_sweep_results.csv"))
    base["lbfgs_cap"] = 5000
    parts = [base] + [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(d, "extra", "runs_*.csv")))]
    df = pd.concat(parts, ignore_index=True)
    df["maxfun"] = df.get("maxfun", pd.Series(15000, index=df.index)).fillna(15000).astype(int)
    df["lbfgs_cap"] = df["lbfgs_cap"].fillna(5000).astype(int)
    df["seed"] = df["seed"].astype(int)
    df = df.drop_duplicates("name", keep="first")  # the T1 0.1% check run repeats the original
    df["standard"] = (df["lbfgs_cap"] == 5000) & (df["seed"] == 99)
    return df


def setting(r):
    bits = []
    if r.lbfgs_cap != 5000:
        bits.append(f"{r.lbfgs_cap // 1000}k L-BFGS cap")
    if r.seed != 99:
        bits.append(f"seed {r.seed}")
    return ", ".join(bits)


def table(df):
    rows = []
    for T in sorted(df["T"].unique()):
        sub = df[df["T"] == T]
        for lev, txt in zip(LEVELS, LEVEL_TXT):
            hit = sub[sub["rel_l2"] <= lev]
            std = hit[hit["standard"]]
            pick = std if len(std) else hit
            if len(pick):
                r = pick.sort_values(["pct_reference", "lbfgs_cap"]).iloc[0]
                rows.append(dict(T=T, level=txt, status="reached", pct=r.pct_reference, n_data=int(r.n_data),
                                 setting=setting(r), rel_l2=r.rel_l2, run=r["name"]))
            else:
                best = sub.dropna(subset=["rel_l2"]).sort_values("rel_l2").iloc[0]
                rows.append(dict(T=T, level=txt, status="failed", pct=best.pct_reference, n_data=int(best.n_data),
                                 setting=f"after {int(best.lbfgs_steps) // 1000}k L-BFGS steps",
                                 rel_l2=best.rel_l2, run=best["name"]))
    return pd.DataFrame(rows)


def cell_text(r):
    if r.status == "failed":
        return f"FAILED\nbest {r.rel_l2:.1e}\nat {r.pct:g}% ({int(r.n_data):,} pts)"
    t = f"{r.pct:g}%\n{int(r.n_data):,} pts"
    return t + (f"\n({r.setting})" if r.setting else "")


def main(d):
    df = load(d)
    tab = table(df)
    reasons_path = os.path.join(d, "extra", "reasons.csv")
    reasons = pd.read_csv(reasons_path) if os.path.exists(reasons_path) else pd.DataFrame(columns=["T", "level", "reason"])
    tab = tab.merge(reasons, on=["T", "level"], how="left")
    tab.to_csv(os.path.join(d, "pct_needed_filled.csv"), index=False)
    df.drop(columns="standard").to_csv(os.path.join(d, "pct_sweep_all_runs.csv"), index=False)

    Ts = sorted(df["T"].unique())
    nref = {T: int(df[df["T"] == T]["n_reference"].iloc[0]) for T in Ts}

    # ---- PNG ----
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    ax.set_xlim(0, len(LEVELS)); ax.set_ylim(len(Ts), 0); ax.axis("off")
    for i, T in enumerate(Ts):
        ax.text(-0.05, i + 0.5, f"T = {T:g}\n({nref[T]:,} ref. pts)", ha="right", va="center", fontsize=12)
        for j, txt in enumerate(LEVEL_TXT):
            r = tab[(tab["T"] == T) & (tab["level"] == txt)].iloc[0]
            if r.status == "failed":
                color, fc = "#f6d5d5", "#a33"
            else:
                shade = np.clip(np.log10(r.pct / 0.1) / np.log10(1000), 0, 1)
                color = plt.cm.Blues(0.2 + 0.5 * shade)
                fc = "black"
                if r.setting:
                    color = plt.cm.Greens(0.25 + 0.4 * shade)
            ax.add_patch(plt.Rectangle((j + 0.015, i + 0.03), 0.97, 0.94, color=color))
            ax.text(j + 0.5, i + 0.5, cell_text(r), ha="center", va="center", fontsize=11, color=fc)
    for j, txt in enumerate(LEVEL_TXT):
        ax.text(j + 0.5, -0.12, txt, ha="center", va="bottom", fontsize=12)
    ax.set_title("One network on [0, T]: smallest tested % of the RK4 reference points that reaches each relative-L2 level\n"
                 "tested 0.01% to 100% of the reference (dt = 1e-4), data spread evenly. Blue: original settings "
                 "(seed 99, 5,000 L-BFGS steps).\nGreen: only reached with a longer L-BFGS cap or another seed. "
                 "Red: no run reached it (best error and the % that got it shown).", fontsize=11, pad=34)
    fig.savefig(os.path.join(d, "pct_needed_table_filled.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Excel ----
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    wb = Workbook(); ws = wb.active; ws.title = "Data needed"
    ws["A1"] = "One network on [0, T]: smallest tested % of the RK4 reference points that reaches each relative-L2 level"
    ws["A2"] = ("Tested 0.01% to 100% of the reference (dt = 1e-4), data spread evenly. Plain cells: original settings "
                "(seed 99, 5,000 L-BFGS steps). Green: needed a longer L-BFGS cap or another seed. Red: FAILED (see Reasons).")
    ws["A1"].font = Font(bold=True, size=12); ws["A2"].font = Font(italic=True, color="555555")
    thin = Side(style="thin", color="BBBBBB"); b = Border(top=thin, bottom=thin, left=thin, right=thin)
    for j, h in enumerate(["T", "Reference points"] + LEVEL_TXT, 1):
        c = ws.cell(4, j, h); c.font = Font(bold=True); c.border = b
        c.fill = PatternFill("solid", fgColor="DDE6F0"); c.alignment = Alignment(horizontal="center")
    for i, T in enumerate(Ts, 5):
        ws.cell(i, 1, T); ws.cell(i, 2, nref[T]).number_format = "#,##0"
        for j in (1, 2):
            ws.cell(i, j).border = b; ws.cell(i, j).alignment = Alignment(horizontal="center", vertical="center")
        for j, txt in enumerate(LEVEL_TXT, 3):
            r = tab[(tab["T"] == T) & (tab["level"] == txt)].iloc[0]
            if r.status == "failed":
                v = f"FAILED (best {r.rel_l2:.1e} at {r.pct:g}%, {int(r.n_data):,} pts, {r.setting})"; fill = "F6D5D5"; font = Font(color="AA3333")
            else:
                v = f"{r.pct:g}% ({int(r.n_data):,} pts)" + (f", {r.setting}" if r.setting else "")
                fill = "D9EAD3" if r.setting else None; font = Font()
            c = ws.cell(i, j, v); c.font = font; c.border = b
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            if fill:
                c.fill = PatternFill("solid", fgColor=fill)
        ws.row_dimensions[i].height = 32
    ws.column_dimensions["A"].width = 8; ws.column_dimensions["B"].width = 17
    for k in "CDEFGH":
        ws.column_dimensions[k].width = 24

    ws2 = wb.create_sheet("Reasons")
    ws2.append(["T", "Level", "Status", "Run used", "Its relative L2", "% of data", "Data points", "L-BFGS cap / seed",
                "Reason (failed cells) / note"])
    for c in ws2[1]:
        c.font = Font(bold=True)
    for r in tab.itertuples():
        ws2.append([r.T, r.level, r.status, r.run, float(r.rel_l2), float(r.pct), int(r.n_data), r.setting or "5k L-BFGS cap, seed 99",
                    r.reason if isinstance(r.reason, str) else (r.setting or "")])
    ws2.column_dimensions["D"].width = 34; ws2.column_dimensions["H"].width = 18; ws2.column_dimensions["I"].width = 110

    ws3 = wb.create_sheet("All runs")
    cols = ["name", "T", "pct_reference", "n_data", "n_reference", "lbfgs_cap", "lbfgs_steps", "seed",
            "rel_l2", "max_abs", "final_loss", "seconds"]
    ws3.append(cols)
    for c in ws3[1]:
        c.font = Font(bold=True)
    for r in df.sort_values(["T", "pct_reference", "lbfgs_cap", "seed"])[cols].itertuples(index=False):
        ws3.append([None if (isinstance(x, float) and np.isnan(x)) else x for x in r])
    ws3.column_dimensions["A"].width = 30
    wb.save(os.path.join(d, "pct_needed_table_filled.xlsx"))
    print(tab[["T", "level", "status", "pct", "setting", "rel_l2"]].to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../results/pct_sweep")
