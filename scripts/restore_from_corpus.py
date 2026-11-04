#!/usr/bin/env python3
"""
restore_from_corpus.py — add word labels and word boundaries back to data/tokens.csv
from a licensed local copy of the Buckeye Corpus.

Boundaries follow the corpus convention: each `.words` time stamp is the END of the
entry, and an entry starts at the end of the preceding one.

Usage:  python3 restore_from_corpus.py --raw /path/to/flat/buckeye --data ../data
Writes data/tokens_restored.csv with the columns `label`, `start` and `end` added.
"""
import argparse
import importlib.util
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("v2", os.path.join(HERE, "11_extract_v2.py"))
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    tok = pd.read_csv(os.path.join(a.data, "tokens.csv"))
    parts = []
    for base, g in tok.groupby("basename", sort=False):
        w = v2.read_xlabel(os.path.join(a.raw, base + ".words"))
        g = g.copy()
        g["label"] = w["label"].to_numpy()[g["words_idx"]]
        g["start"] = w["start"].to_numpy()[g["words_idx"]]
        g["end"] = w["end"].to_numpy()[g["words_idx"]]
        parts.append(g)
    out = pd.concat(parts).loc[tok.index]
    out.to_csv(os.path.join(a.data, "tokens_restored.csv"), index=False)
    print(f"Restored {len(out)} tokens -> {os.path.join(a.data, 'tokens_restored.csv')}")


if __name__ == "__main__":
    main()
