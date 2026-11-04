#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_f0_supplementary.py — F0可用性 < 50% トークンの集計（補足分析）

Colab notebook `re-encoding_after_self-repair.ipynb`（最終セル）のローカル移植版。
data_for_r.csv の f0_avail 列（= f0_k / f0_n）が 0.5 未満のトークン割合を条件別に集計。

使い方:  python3 04_f0_supplementary.py  [--out ../output]
"""
from __future__ import annotations
import argparse
import os

import pandas as pd


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(root, "output"))
    args = ap.parse_args()

    pairs_df = pd.read_csv(os.path.join(args.out, "data_for_r.csv"))
    pairs_df["lt50"] = pairs_df["f0_avail"] < 0.5
    f0_stats = pairs_df.groupby("condition")["lt50"].agg(["sum", "mean", "count"])
    f0_stats["percent"] = f0_stats["mean"] * 100

    print("Tokens with F0 Availability < 50%:")
    print("-" * 40)
    for cond, label in [("first", "First Production:      "),
                        ("repeat", "Post-Repair Repetition:")]:
        if cond in f0_stats.index:
            n = int(f0_stats.loc[cond, "sum"])
            pct = f0_stats.loc[cond, "percent"]
            print(f"{label} N = {n} ({pct:.1f}%)")
    print("-" * 40)


if __name__ == "__main__":
    main()
