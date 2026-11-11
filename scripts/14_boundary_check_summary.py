#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14_boundary_check_summary.py — 語区間の目視確認結果の集計と §2.2 用の 1 文の下書き

入力:
  --results  結果 CSV（既定: 14_boundary_check_serve.py が自動保存する boundary_check/results.csv）
             （item_id, word, onset, offset, identity, note, timestamp）
  --key      boundary_check/key.csv（item_id と条件の対応表。判定が終わってから開く）
出力:
  標準出力に全体・条件別（first/repeat/fluent）の件数と、英語 1 文の下書き。
  --out を指定すると、結合した表（結果 + 条件）を CSV に保存する。
  結合表にはコーパス由来の時刻・話者を含めない（item_id・条件・判定・メモのみ）。

使い方:  python3 scripts/14_boundary_check_summary.py
"""
from __future__ import annotations
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
       10: "ten", 11: "eleven", 12: "twelve"}
COND_LABEL = [("first", "first productions"), ("repeat", "repetitions"), ("fluent", "fluent tokens")]


def main() -> None:
    root = os.path.dirname(HERE)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=os.path.join(root, "boundary_check", "results.csv"))
    ap.add_argument("--key", default=os.path.join(root, "boundary_check", "key.csv"))
    ap.add_argument("--out", default=None, help="結合表の保存先（省略可）")
    args = ap.parse_args()

    if not os.path.isfile(args.results):
        sys.exit(f"結果ファイルがまだない: {os.path.relpath(args.results, root)}\n"
                 "判定は python3 scripts/14_boundary_check_serve.py で開いたツールで行う（1件判定すると自動で作られる）。")
    res = pd.read_csv(args.results, encoding="utf-8-sig", keep_default_na=False)
    key = pd.read_csv(args.key)
    extra = set(res.item_id) - set(key.item_id)
    if extra:
        sys.exit(f"key.csv にない item_id が結果にあります: {sorted(extra)[:5]}")
    d = key.merge(res, on="item_id", how="left")
    cols = ["onset", "offset", "identity"]
    done = d[cols].notna().all(axis=1) & d[cols].ne("").all(axis=1)
    if not done.all():
        print(f"注意: 判定が未完の項目が {int((~done).sum())} 件あります。集計から除外します。")
        d = d[done]
    d = d.copy()
    d["onset_ok"] = d.onset.eq("correct")
    d["offset_ok"] = d.offset.eq("correct")
    d["id_ok"] = d.identity.eq("correct")
    d["edges_ok"] = d.onset_ok & d.offset_ok
    d["all_ok"] = d.edges_ok & d.id_ok

    def tab(g: pd.DataFrame) -> dict:
        return {"n": len(g), "onset_ok": int(g.onset_ok.sum()), "offset_ok": int(g.offset_ok.sum()),
                "both_edges_ok": int(g.edges_ok.sum()), "id_ok": int(g.id_ok.sum()), "all_ok": int(g.all_ok.sum())}

    rows = {"all": tab(d)}
    for c, _ in COND_LABEL:
        rows[c] = tab(d[d.condition == c])
    print(pd.DataFrame(rows).T.to_string())
    bad = d[~d.all_ok]
    if len(bad):
        print("\nずれ・別の語と判定された項目:")
        for r in bad.itertuples():
            print(f"  {r.item_id} [{r.condition}] onset={r.onset} offset={r.offset} identity={r.identity}"
                  + (f"  メモ: {r.note}" if str(r.note) else ""))

    n = len(d)
    parts = [f"{rows[c]['n']} {lab}" for c, lab in COND_LABEL if rows[c]["n"]]
    parts_s = ", ".join(parts[:-1]) + (", and " if len(parts) > 2 else " and ") + parts[-1] if len(parts) > 1 else parts[0]
    a = rows["all"]
    sent = (f"To verify the corrected boundaries, the author inspected {n} randomly selected tokens ({parts_s}) "
            f"by listening to each word interval with 500 ms of context and the waveform; "
            f"{a['all_ok']} of {n} word intervals were judged correct at both edges and as the intended word "
            f"(onset: {a['onset_ok']}/{n}; offset: {a['offset_ok']}/{n}; word identity: {a['id_ok']}/{n}).")
    print("\n§2.2 下書き（要確認。確認者・手順の表現は実際に合わせて直す）:\n" + sent)
    if args.out:
        d[["item_id", "condition", "onset", "offset", "identity", "note", "timestamp"]].to_csv(args.out, index=False)
        print(f"\n結合表を保存: {args.out}")


if __name__ == "__main__":
    main()
