#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_figures_v2.py — R1 改稿用の図（output_v2/ から作成）

  Figure1_pairs.png       完全ペアの first vs repeat（log 持続時間・話者中心化強度）、同一線つき
  Figure2_estimates.png   モデル推定値: repeat−first（90% CI、同等性域）と、first/repeat−fluent（95% CI）
  FigureS1_f0avail.png    条件別 F0 可用性（箱ひげ + 個別点）
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output_v2")
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})

w = pd.read_csv(os.path.join(OUT, "pairs_wide_v2.csv"))
est = pd.read_csv(os.path.join(OUT, "estimates_v2.csv")).set_index("label")
tok = pd.read_csv(os.path.join(OUT, "tokens.csv"))

# ── Figure 1 ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=True)
panels = [("log_dur.first", "log_dur.repeat", "Log word duration (ln s)", "A. Duration"),
          ("int_c.first", "int_c.repeat", "Speaker-centred intensity (dB)", "B. Intensity")]
for ax, (x, y, lab, title) in zip(axes, panels):
    d = w[[x, y]].dropna()
    ax.scatter(d[x], d[y], s=10, color=INK, alpha=0.35, linewidths=0)
    lo, hi = np.nanpercentile(np.r_[d[x], d[y]], [0.5, 99.5])
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1, ls="--")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad); ax.set_aspect("equal")
    ax.set_xlabel(f"First production\n{lab}"); ax.set_ylabel(f"Post-repair repetition\n{lab}")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.grid(color=GRID, lw=0.5)
    r = np.corrcoef(d[x], d[y])[0, 1]
    ax.text(0.03, 0.97, f"n = {len(d)} pairs\nr = {r:.2f}", transform=ax.transAxes,
            va="top", fontsize=9, color=INK)
fig.savefig(os.path.join(OUT, "Figure1_pairs.png"), dpi=300)
plt.close(fig)

# ── Figure 2 ─────────────────────────────────────────────────────
def pct(x):
    return 100 * (np.exp(x) - 1)

rows = [  # (panel, y-label, estimate label, transform, CI columns)
    ("dur", "Repeat − first", "DUR repeat-first (all tokens)", "90"),
    ("dur", "First − fluent", "DUR vs fluent: conditionfirst", "95"),
    ("dur", "Repeat − fluent", "DUR vs fluent: conditionrepeat", "95"),
    ("int", "Repeat − first", "INT repeat-first (all tokens)", "90"),
    ("int", "First − fluent", "INT vs fluent: conditionfirst", "95"),
    ("int", "Repeat − fluent", "INT vs fluent: conditionrepeat", "95"),
]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True, sharey=True)
spec = {"dur": ("Difference in word duration (%)", pct, 10.0, "A. Duration"),
        "int": ("Difference in intensity (dB)", lambda v: v, 1.0, "B. Intensity")}
for ax, key in zip(axes, ["dur", "int"]):
    xlabel, f, sesoi, title = spec[key]
    sub = [r for r in rows if r[0] == key]
    ax.axvspan(-sesoi, sesoi, color=GRID, alpha=0.6, lw=0)
    ax.axvline(0, color=MUTED, lw=1)
    for i, (_, lab, el, ci) in enumerate(sub):
        e = est.loc[el]
        b, lo, hi = f(e["b"]), f(e[f"lo{ci}"]), f(e[f"hi{ci}"])
        yy = len(sub) - 1 - i
        marker = "D" if i == 0 else "o"
        ax.plot([lo, hi], [yy, yy], color=INK, lw=2, solid_capstyle="round")
        ax.plot(b, yy, marker=marker, ms=8, color=INK if i else "white", mec=INK, mew=1.5)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels([r[1] for r in sub][::-1])
    ax.set_xlabel(xlabel); ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.text(-sesoi + 0.02 * sesoi, len(sub) - 0.45, f"equivalence bounds ±{sesoi:g}", ha="left", fontsize=8, color=MUTED)
    ax.set_ylim(-0.6, len(sub) - 0.2)
    ax.grid(axis="x", color=GRID, lw=0.5)
fig.savefig(os.path.join(OUT, "Figure2_estimates.png"), dpi=300)
plt.close(fig)

# ── Figure S1 ────────────────────────────────────────────────────
t = tok[(tok.keep_int == 1) & (tok.f0_n > 0)].copy()
t["f0_avail"] = t.f0_k / t.f0_n
order = [("fluent", "Fluent baseline"), ("first", "First production"), ("repeat", "Post-repair repetition")]
fig, ax = plt.subplots(figsize=(5.0, 3.4), constrained_layout=True)
data = [t.loc[t.condition == c, "f0_avail"].values for c, _ in order]
ax.boxplot(data, widths=0.5, showfliers=False, medianprops={"color": INK, "lw": 2},
           boxprops={"color": INK}, whiskerprops={"color": INK}, capprops={"color": INK})
rng = np.random.default_rng(1)
for i, d in enumerate(data, start=1):
    dd = d if len(d) <= 1000 else rng.choice(d, 1000, replace=False)
    ax.scatter(i + rng.uniform(-0.15, 0.15, len(dd)), dd, s=4, color=MUTED, alpha=0.25, linewidths=0)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels([f"{l}\n(n = {len(d)})" for (_, l), d in zip(order, data)])
ax.set_ylabel("F0 availability\n(proportion of voiced frames)")
ax.grid(axis="y", color=GRID, lw=0.5)
fig.savefig(os.path.join(OUT, "FigureS1_f0avail.png"), dpi=300)
plt.close(fig)
print("Saved Figure1_pairs.png, Figure2_estimates.png, FigureS1_f0avail.png")
