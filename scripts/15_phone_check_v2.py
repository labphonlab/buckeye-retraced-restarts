#!/usr/bin/env python3
"""
15_phone_check_v2.py — 直後型事象で R が長いのは境界のずれか、実際の引き延ばしか（DECISIONS.md D20・D21）

12c_abut_sensitivity_v2.R で、R が cutoff 断片の直後に始まる事象（gap_cut_to_repeat == 0）では
R が F より約 12% 長く約 1 dB 弱かった。Buckeye の .phones で F・R・断片の音素を引き、次を比べる。
  - R の語頭の音素が F の語頭の音素より長いか（断片の末尾を取り込んでいれば語頭だけが伸びる）
  - 語頭以外の部分も R で長いか（実際の引き延ばしなら語全体または語末が伸びる）
  - R の音素列が F と同じか（断片の音を語頭に余分に含んでいないか）
  - 断片が直後型で短いか（R の語頭が早すぎれば、その分だけ断片が短く切られる）

入力:  output_v2/{tokens,events}.csv、raw/*.words・*.phones
出力:  output_v2/phone_check_v2.txt（集計のみ。音素ラベル・コーパス時刻は書き出さない）
使い方:  python3 scripts/15_phone_check_v2.py
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output_v2")
RAW = os.path.join(ROOT, "raw")
_spec = importlib.util.spec_from_file_location("x11", os.path.join(HERE, "11_extract_v2.py"))
x11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(x11)  # read_xlabel を共有（時刻 = 終了時刻）
EPS = 1e-4


def phones_in(ph: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    mid = (ph.start + ph.end) / 2
    return ph[(mid > t0) & (mid < t1)]


def main() -> None:
    tok = pd.read_csv(os.path.join(OUT, "tokens.csv"))
    ev = pd.read_csv(os.path.join(OUT, "events.csv"))
    ev = ev[ev.status == "ok"].set_index("event_id")
    et = tok[(tok.condition != "fluent") & (tok.keep_dur == 1)]
    pairs = et.pivot_table(index=["event_id", "basename"], columns="condition",
                           values=["start", "end"]).dropna()
    rows, cache = [], {}
    for (eid, base), r in pairs.iterrows():
        if base not in cache:
            cache[base] = (x11.read_xlabel(os.path.join(RAW, f"{base}.phones")),
                           x11.read_xlabel(os.path.join(RAW, f"{base}.words")))
        ph, wd = cache[base]
        rec = {"event_id": eid, "abut": int(ev.loc[eid, "gap_cut_to_repeat"] == 0)}
        seqs = {}
        for c in ("first", "repeat"):
            p = phones_in(ph, r[("start", c)], r[("end", c)])
            seqs[c] = list(p.label)
            d = (p.end - p.start).to_numpy()
            rec[f"{c}_n"] = len(d)
            rec[f"{c}_word"] = r[("end", c)] - r[("start", c)]
            rec[f"{c}_first"] = d[0] if len(d) else np.nan
            rec[f"{c}_last"] = d[-1] if len(d) else np.nan
            rec[f"{c}_rest"] = d[1:].sum() if len(d) > 1 else np.nan
            rec[f"{c}_aligned"] = int(len(d) > 0 and abs(p.start.iloc[0] - r[("start", c)]) < EPS
                                      and abs(p.end.iloc[-1] - r[("end", c)]) < EPS)
        rec["same_seq"] = int(seqs["first"] == seqs["repeat"])
        rec["extra_initial"] = int(len(seqs["repeat"]) > len(seqs["first"])
                                   and seqs["repeat"][-len(seqs["first"]):] == seqs["first"])
        # 断片: F の後で最初の cutoff ラベル
        cut = wd[(wd.start >= r[("end", "first")] - EPS) & wd.label.str.startswith("<CUTOFF")].head(1)
        if len(cut):
            fp = phones_in(ph, cut.start.iloc[0], cut.end.iloc[0])
            rec["frag_dur"] = cut.end.iloc[0] - cut.start.iloc[0]
            rec["frag_last"] = (fp.end - fp.start).iloc[-1] if len(fp) else np.nan
            rec["frag_n"] = len(fp)
        rows.append(rec)
    d = pd.DataFrame(rows)
    d["speaker"] = d.event_id.map(ev.speaker)
    d["word"] = d.event_id.map(ev.word)
    for part in ("word", "first", "rest", "last"):
        d[f"lr_{part}"] = np.log(d[f"repeat_{part}"] / d[f"first_{part}"])

    lines = ["=== 15_phone_check_v2.py ===",
             f"events with both tokens: {len(d)} (abutting {d.abut.sum()}, gap {(1 - d.abut).sum()})",
             f"word edges coincide with phone edges: F {d.first_aligned.mean():.3f}, R {d.repeat_aligned.mean():.3f}", ""]
    g = d.groupby("abut")
    lines.append("Medians by abut (0 = gap, 1 = abutting); durations in ms, ratios R/F")
    tab = pd.DataFrame({
        "n": g.size(),
        "F_word": g.first_word.median() * 1000, "R_word": g.repeat_word.median() * 1000,
        "F_first": g.first_first.median() * 1000, "R_first": g.repeat_first.median() * 1000,
        "F_rest": g.first_rest.median() * 1000, "R_rest": g.repeat_rest.median() * 1000,
        "F_last": g.first_last.median() * 1000, "R_last": g.repeat_last.median() * 1000,
        "frag_dur": g.frag_dur.median() * 1000, "frag_last": g.frag_last.median() * 1000,
        "frag_n": g.frag_n.median(),
        "same_seq": g.same_seq.mean(), "extra_initial": g.extra_initial.mean(),
        "R_more_phones": g.apply(lambda x: (x.repeat_n > x.first_n).mean()),
        "R_fewer_phones": g.apply(lambda x: (x.repeat_n < x.first_n).mean()),
    })
    lines.append(tab.round(3).T.to_string())
    lines.append("")
    lines.append("Geometric-mean R/F ratio by abut, and abut effect on log R/F "
                 "(mixed model, random intercepts for speaker; word as variance component)")
    for part in ("word", "first", "rest", "last"):
        sub = d.dropna(subset=[f"lr_{part}"])
        sub = sub[np.isfinite(sub[f"lr_{part}"])]
        gm = sub.groupby("abut")[f"lr_{part}"].mean().map(np.exp)
        m = smf.mixedlm(f"lr_{part} ~ abut", sub, groups=sub.speaker,
                        vc_formula={"word": "0 + C(word)"}).fit(reml=True)
        b, se, p = m.params["abut"], m.bse["abut"], m.pvalues["abut"]
        lines.append(f"  {part:5s} n={len(sub):3d}  gap {gm.get(0, np.nan):.3f}  abut {gm.get(1, np.nan):.3f}  "
                     f"abut effect b={b:+.3f} SE={se:.3f} p={p:.4f} (x{np.exp(b):.3f})")
    # 音素列が同じ事象だけで同じ比較（余分な音素を含む対を除く）
    lines.append("")
    lines.append("Same comparison restricted to events whose F and R phone sequences are identical")
    same = d[d.same_seq == 1]
    for part in ("word", "first", "rest", "last"):
        sub = same.dropna(subset=[f"lr_{part}"])
        sub = sub[np.isfinite(sub[f"lr_{part}"])]
        gm = sub.groupby("abut")[f"lr_{part}"].mean().map(np.exp)
        lines.append(f"  {part:5s} n={len(sub):3d}  gap {gm.get(0, np.nan):.3f}  abut {gm.get(1, np.nan):.3f}")
    # 1音素語（I, a など）は語頭 = 語全体なので、2音素以上の語だけで語頭と残りを分ける
    lines.append("")
    lines.append("Multi-phone words only (F and R both > 1 phone): geometric-mean R/F ratio")
    multi = d[(d.first_n > 1) & (d.repeat_n > 1)]
    for part in ("word", "first", "rest", "last"):
        gm = multi.groupby("abut")[f"lr_{part}"].mean().map(np.exp)
        lines.append(f"  {part:5s} n={len(multi):3d}  gap {gm.get(0, np.nan):.3f}  abut {gm.get(1, np.nan):.3f}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "phone_check_v2.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
