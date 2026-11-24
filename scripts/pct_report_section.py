"""Report Section 21: filling in the data-needed table.

Builds two graphs, the section's pages (reportlab) and a new contents page, then
writes a new Lorenz_PINN_Report.pdf: old page 1, new contents, old pages 3+, new
section. The previous report is kept as Lorenz_PINN_Report_before_section21.pdf.

    python pct_report_section.py /mnt/project-files/results
"""
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle, PageBreak)

RES = sys.argv[1] if len(sys.argv) > 1 else "../results"
PCT = os.path.join(RES, "pct_sweep")
EXTRA = os.path.join(PCT, "extra")
REPORT = os.path.join(RES, "Lorenz_PINN_Report.pdf")
BACKUP = os.path.join(RES, "Lorenz_PINN_Report_before_section21.pdf")
WORK = os.environ.get("WORK", "/tmp")
TITLE = "Teaching a Neural Network the Lorenz System"
BLUE = colors.HexColor("#1f5fa8")

body = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=15, spaceAfter=6)
h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17, leading=22, spaceAfter=10)
h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13, leading=17, spaceBefore=10, spaceAfter=6)
cap = ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=8.5, leading=11, alignment=TA_CENTER,
                     textColor=colors.HexColor("#555555"), spaceAfter=10)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=2, spaceAfter=4)
cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.5, leading=10.5)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
boxh = ParagraphStyle("boxh", fontName="Helvetica-Bold", fontSize=9.5, textColor=BLUE, spaceAfter=4)


def sci(x, d=1):
    """2.3e-05 -> 2.3 × 10<super>-5</super>"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "no result"
    m, e = f"{x:.{d}e}".split("e")
    return f"{m} × 10<super>{int(e)}</super>"


def box(title, text):
    t = Table([[[Paragraph(title, boxh), Paragraph(text, body)]]], colWidths=[6.5 * inch])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdf0e6")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, -1), (-1, -1), 8)]))
    return t


def grid(rows, widths, header=True):
    data = [[Paragraph(str(c), cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")))
    t.setStyle(TableStyle(st))
    return t


def B(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


# ---------------------------------------------------------------- data
def load():
    base = pd.read_csv(os.path.join(PCT, "pct_sweep_results.csv"))
    base["lbfgs_cap"] = 5000
    ex = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(EXTRA, "runs_*.csv")))],
                   ignore_index=True)
    ex["lbfgs_cap"] = ex["lbfgs_cap"].astype(int)
    ex["seed"] = ex["seed"].astype(int)
    ex = ex[ex["name"] != "T1_pct0.1"]  # the reproduction check, reported separately
    return base, ex


def plot_data(base, ex):
    std = pd.concat([base, ex[(ex["lbfgs_cap"] == 5000) & (ex["seed"] == 99)]])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for T, c in zip([1, 1.5, 2, 2.5, 3], ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#d62728"]):
        d = std[std["T"] == T].sort_values("pct_reference").dropna(subset=["rel_l2"])
        ax.plot(d["pct_reference"], d["rel_l2"], "o-", color=c, label=f"T = {T:g}", ms=4)
    for lev in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        ax.axhline(lev, color="#999999", lw=0.6, ls="--" if lev == 1e-4 else ":")
    ax.axvspan(30, 150, color="#f2f2f2", zorder=0)
    ax.text(70, 2.5, "new: 50%, 100%", ha="center", fontsize=8, color="#666666")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(3e-7, 5)
    ax.set_xticks([0.01, 0.03, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["0.01", "0.03", "0.1", "0.25", "0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("data used: % of the RK4 reference points on [0, T]")
    ax.set_ylabel("relative L2 over [0, T]")
    ax.set_title("One network, original settings (seed 99, 5,000 L-BFGS steps), now up to 100% of the points",
                 fontsize=10)
    ax.legend(fontsize=8, ncol=5, loc="lower center")
    path = os.path.join(WORK, "s21_data.png")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    return path


def plot_steps():
    fig, ax = plt.subplots(figsize=(9, 4.4))
    runs = [("T1.5_pct0.25_lbfgs60000_maxfun120000", "T = 1.5, 0.25%", "#2ca02c"),
            ("T2_pct2_lbfgs60000_maxfun120000", "T = 2, 2%", "#9467bd"),
            ("T2_pct0.25_lbfgs60000_maxfun120000", "T = 2, 0.25%", "#c5b0d5"),
            ("T2.5_pct1_lbfgs100000_maxfun200000", "T = 2.5, 1%", "#ff7f0e"),
            ("T3_pct10_lbfgs200000_maxfun400000", "T = 3, 10%", "#d62728")]
    for name, lab, c in runs:
        f = os.path.join(EXTRA, "checkpoints", name + "_progress.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f, names=["steps", "rel_l2", "time"]).drop_duplicates("steps", keep="last")
        ax.plot(d["steps"] / 1000, d["rel_l2"], "o-", ms=3, color=c, label=lab)
    for lev in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        ax.axhline(lev, color="#999999", lw=0.6, ls="--" if lev == 1e-4 else ":")
    ax.set_yscale("log"); ax.set_xlabel("L-BFGS steps (thousands), after 10,000 Adam epochs")
    ax.set_ylabel("relative L2 over [0, T]")
    ax.set_title("Error as L-BFGS keeps going (one checkpoint every 5,000 steps)", fontsize=10)
    ax.legend(fontsize=8)
    path = os.path.join(WORK, "s21_steps.png")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    return path


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
    from section21_text import story_for, SECTION_TITLE  # the words live in their own file
    base, ex = load()
    figs = dict(data=plot_data(base, ex), steps=plot_steps(),
                table=os.path.join(PCT, "pct_needed_table_filled.png"))
    filled = pd.read_csv(os.path.join(PCT, "pct_needed_filled.csv"))

    old = PdfReader(REPORT if not os.path.exists(BACKUP) else BACKUP)
    if not os.path.exists(BACKUP):
        shutil.copy(REPORT, BACKUP)
    first = len(old.pages) + 1
    sec = os.path.join(WORK, "section21.pdf")
    build_doc(sec, story_for(base, ex, filled, figs, dict(h1=h1, h2=h2, body=body, cap=cap, B=B, box=box,
                                                           grid=grid, sci=sci, Image=Image, Spacer=Spacer,
                                                           Paragraph=Paragraph, KeepTogether=KeepTogether,
                                                           PageBreak=PageBreak, inch=inch)), first)

    lines = old.pages[1].extract_text().splitlines()
    titles = [l.split(". ", 1)[1] for l in lines if l[:1].isdigit() and ". " in l]
    titles = [t for t in titles if not t.startswith(SECTION_TITLE)] + [SECTION_TITLE]
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
    print(f"report: {len(w.pages)} pages, section 21 starts on page {first}")


if __name__ == "__main__":
    main()
