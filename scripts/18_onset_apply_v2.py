#!/usr/bin/env python3
"""
18_onset_apply_v2.py — 付け直した語頭を当てはめて、直後型の R を測り直す（DECISIONS.md D23）

入力:  onset_check/results.csv（16_onset_serve.py が保存）, onset_check/key.csv, output_v2/tokens.csv, raw/
出力:
  output_v2/onset_agreement_v2.txt  群ごとの「付け直し − コーパス」のずれ（ms）の要約（時刻・ラベルなし）
  output_v2/tokens_onsetfix_v2.csv      直後型 R の語頭を付け直した位置に替えた tokens 表
  output_v2/tokens_onsetfix_cal_v2.csv  同上。ただし対照（間のある R と F）のずれの中央値を差し引いて較正した位置
  どちらも、直後型 R のうち「付けられない」とした項目は keep_dur = keep_int = 0（分析から除く）。
  持続時間・強度・F0 は 11_extract_v2.py の measure() で測り直し、keep_dur・keep_int も同じ基準で付け直す。
  直後型 R 以外のトークン（対照を含む）は変えない。
中間集計（DECISIONS.md D24）: 未判定の項目が残っているときは tokens 表を作らず、
  output_v2/onset_agreement_interim_v2.txt に、判定済みの項目だけのずれの要約と、
  直後型 R と対照の差（語の長さの対数比、ブートストラップ 95% CI）と、続けるかどうかの判定を書く。
使い方:  python3 scripts/18_onset_apply_v2.py [--results 別の results.csv]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import parselmouth

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output_v2")
KIT = os.path.join(ROOT, "onset_check")
_spec = importlib.util.spec_from_file_location("x11", os.path.join(HERE, "11_extract_v2.py"))
x11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(x11)
TOL = 1e-6
STOP_BOUND = 0.05  # D24: 差の 95% CI が ±0.05（約 ±5%）に収まれば付け直しを打ち切る
MIN_INTERIM = 120  # D24: 中間集計は判定済み 120 項目以上で行う


def summarize(d: pd.DataFrame) -> str:
    rows = []
    for g, x in d.groupby("group"):
        s = x.shift_ms.dropna()
        rows.append({"group": g, "items": len(x), "marked": int((x.status == "marked").sum()),
                     "unsure": int((x.status == "unsure").sum()), "cannot": int((x.status == "cannot").sum()),
                     "median_ms": s.median(), "mean_ms": s.mean(), "sd_ms": s.std(),
                     "q25_ms": s.quantile(.25), "q75_ms": s.quantile(.75),
                     "later_gt20": (s > 20).mean(), "earlier_gt20": (s < -20).mean(), "abs_le10": (s.abs() <= 10).mean()})
    return pd.DataFrame(rows).set_index("group").round(3).T.to_string()


def contrast(d: pd.DataFrame, n_boot: int = 10000, seed: int = 20261106) -> tuple[str, float, float, float]:
    """直後型 R と対照（間のある R と F）の差。指標は付け直しによる語の長さの対数比
    log((end − new_start) / (end − orig_start)) の平均。直後型 R の効果（log 0.110）と同じ尺度。"""
    x = d[d.onset_s.notna()].copy()
    late = x.new_start >= x.orig_end - 0.01  # remeasure() と同じ基準：語頭が語末を越えたら長さを定義できないので除く
    n_late = int(late.sum())
    x = x[~late]
    x["lr"] = np.log((x.orig_end - x.new_start) / (x.orig_end - x.orig_start))
    a = x.lr[x.group == "repeat_abut"].to_numpy()
    c = x.lr[x.group != "repeat_abut"].to_numpy()
    if len(a) < 2 or len(c) < 2:
        return "contrast: too few judged items", np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(a, len(a)).mean() - rng.choice(c, len(c)).mean() for _ in range(n_boot)])
    diff, lo, hi = a.mean() - c.mean(), *np.quantile(boot, [.025, .975])
    txt = (f"log duration ratio (re-marked / corpus), mean: abutting R {a.mean():+.3f} (n {len(a)}), "
           f"controls {c.mean():+.3f} (n {len(c)})\n"
           f"difference abutting R − controls: {diff:+.3f} [95% bootstrap CI {lo:+.3f}, {hi:+.3f}] "
           f"(≈ {100 * np.expm1(diff):+.1f}%; abutting R − F effect in the corpus: log 0.110 ≈ +11.6%)\n"
           f"excluded: re-marked onset at or after the corpus word end (within 10 ms): {n_late}")
    return txt, diff, lo, hi


def remeasure(tok: pd.DataFrame, new_start: pd.Series) -> pd.DataFrame:
    """new_start（tokens.csv の行番号 → 新しい語頭、NaN は除外）で直後型 R を測り直す。"""
    tok = tok.copy()
    cache = {}
    for i, t0 in new_start.items():
        r = tok.loc[i]
        if np.isnan(t0) or t0 >= r.end - 0.01:
            tok.loc[i, ["keep_dur", "keep_int"]] = 0
            continue
        if r.basename not in cache:
            cache[r.basename] = (parselmouth.Sound(os.path.join(ROOT, "raw", f"{r.basename}.wav")),
                                 x11.read_xlabel(os.path.join(ROOT, "raw", f"{r.basename}.phones")))
        snd, ph = cache[r.basename]
        m = x11.measure(snd, ph, t0, r.end)
        tok.loc[i, "start"] = t0
        tok.loc[i, "duration"] = r.end - t0
        for k, v in m.items():
            tok.loc[i, k] = v
        dur = r.end - t0
        tok.loc[i, "keep_dur"] = int(0.05 <= dur <= 2.0)
        tok.loc[i, "keep_int"] = int(tok.loc[i, "keep_dur"] == 1 and tok.loc[i, "intensity"] >= 20)
    return tok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--results", default=os.path.join(KIT, "results.csv"))
    args = ap.parse_args()
    res_p = args.results
    if not os.path.isfile(res_p):
        sys.exit("onset_check/results.csv がまだない。python3 scripts/16_onset_serve.py で語頭を付けてから実行する。")
    res = pd.read_csv(res_p, keep_default_na=False)
    key = pd.read_csv(os.path.join(KIT, "key.csv"))
    d = key.merge(res[["item_id", "onset_s", "status"]], on="item_id", how="left")
    d["status"] = d.status.fillna("")
    d["onset_s"] = pd.to_numeric(d.onset_s, errors="coerce")
    d.loc[d.status == "cannot", "onset_s"] = np.nan
    d["new_start"] = d.clip_start + d.onset_s
    d["shift_ms"] = (d.new_start - d.orig_start) * 1000
    todo = (d.status == "").sum()
    ctrl = d[d.group.isin(["repeat_gap", "first"])].shift_ms.dropna()
    cal = float(ctrl.median()) if len(ctrl) else 0.0
    ctxt, _, lo, hi = contrast(d)

    if todo:  # 中間集計（D24）: tokens 表は作らない
        done = d[d.status != ""]
        n_cannot = done.groupby("group").status.apply(lambda s: int((s == "cannot").sum())).to_dict()
        if len(done) < MIN_INTERIM:
            verdict = f"not yet: {len(done)} judged, interim needs {MIN_INTERIM}"
        elif -STOP_BOUND <= lo and hi <= STOP_BOUND:
            verdict = (f"STOP: 95% CI within ±{STOP_BOUND}; re-marking does not change abutting R materially. "
                       "Corpus onsets are retained; report the judged items as a validation sample.")
        else:
            verdict = f"CONTINUE: 95% CI not within ±{STOP_BOUND}; mark all remaining items."
        lines = ["=== 18_onset_apply_v2.py (interim, D24) ===",
                 f"items {len(d)}, judged {len(done)}, not yet judged {todo}",
                 "shift = re-marked onset − corpus onset (ms); positive = re-marked later", "",
                 summarize(done), "",
                 f"'cannot' by group: {n_cannot}", ctxt, "", f"decision: {verdict}"]
        txt = "\n".join(lines)
        with open(os.path.join(OUT, "onset_agreement_interim_v2.txt"), "w") as f:
            f.write(txt + "\n")
        print(txt)
        return

    tok = pd.read_csv(os.path.join(OUT, "tokens.csv"))
    ab = d[d.group == "repeat_abut"]
    idx = {}
    for r in ab.itertuples():
        hit = tok.index[(tok.basename == r.basename) & (tok.event_id == r.event_id) & (tok.condition == "repeat")
                        & ((tok.start - r.orig_start).abs() < TOL)]
        if len(hit) != 1:
            sys.exit(f"tokens.csv で {r.item_id} の行が一意に見つからない")
        idx[hit[0]] = r
    ns = pd.Series({i: r.new_start for i, r in idx.items()})
    ns_cal = ns - cal / 1000
    remeasure(tok, ns).to_csv(os.path.join(OUT, "tokens_onsetfix_v2.csv"), index=False)
    remeasure(tok, ns_cal).to_csv(os.path.join(OUT, "tokens_onsetfix_cal_v2.csv"), index=False)

    lines = ["=== 18_onset_apply_v2.py ===",
             f"items {len(d)}, not yet judged {todo}",
             "shift = re-marked onset − corpus onset (ms); positive = re-marked later", "",
             summarize(d), "",
             ctxt, "",
             f"calibration (median shift of control items, gap R + F): {cal:.1f} ms",
             f"abutting R re-measured: {ns.notna().sum()}, excluded as 'cannot': {ns.isna().sum()}"]
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "onset_agreement_v2.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
