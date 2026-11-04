#!/usr/bin/env python3
"""
make_public_tables.py — build the shareable token and event tables.

The Buckeye licence does not allow redistribution of the corpus content, so the
public tables omit the corpus time stamps, word labels and cutoff labels. Each
token is identified instead by its recording (`basename`) and its 0-based entry
index in that recording's `.words` file (`words_idx`). Word types are replaced by
stable codes (`w001` ...). `restore_from_corpus.py` puts the labels and times back
for anyone with a licensed copy of the corpus.

Usage:  python3 make_public_tables.py --raw ../raw --src ../output_v2 --out ../data
"""
import argparse
import importlib.util
import os
import subprocess

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("v2", os.path.join(HERE, "11_extract_v2.py"))
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = pd.read_csv(os.path.join(a.src, "tokens.csv"))
    ev = pd.read_csv(os.path.join(a.src, "events.csv"))

    idx = np.full(len(tok), -1)
    for base, g in tok.groupby("basename"):
        w = v2.read_xlabel(os.path.join(a.raw, base + ".words"))
        for i, end in zip(g.index, g["end"]):
            hit = np.flatnonzero(np.abs(w["end"].to_numpy() - end) < 1e-6)
            assert len(hit) == 1, (base, end, hit)
            idx[tok.index.get_loc(i)] = hit[0]
    tok.insert(0, "words_idx", idx)

    # Codes follow R's collation order of the original labels, so that factor levels, and
    # therefore lme4's optimisation path, are the same as in the original analysis.
    words = sorted(set(tok.word.dropna()) | set(ev.word.dropna()))
    r = subprocess.run(["Rscript", "-e", "cat(sort(readLines(file('stdin'))), sep='\\n')"],
                       input="\n".join(words), capture_output=True, text=True, check=True)
    codes = {w: f"w{i + 1:03d}" for i, w in enumerate(r.stdout.splitlines())}
    assert set(codes) == set(words)
    tok["word"] = tok["word"].map(codes)
    ev["word"] = ev["word"].map(codes)
    tok = tok.drop(columns=["start", "end"])
    ev = ev.drop(columns=["cut_label"])
    os.makedirs(a.out, exist_ok=True)
    tok.to_csv(os.path.join(a.out, "tokens.csv"), index=False, float_format="%.17g")
    ev.to_csv(os.path.join(a.out, "events.csv"), index=False, float_format="%.17g")
    print(f"tokens: {len(tok)}  events: {len(ev)}  word codes: {len(codes)}")


if __name__ == "__main__":
    main()
