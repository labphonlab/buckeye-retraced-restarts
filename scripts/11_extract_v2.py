#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_extract_v2.py — 修正版: 同語反復ペアと流暢ベースラインの抽出・音響測定（R1 改稿用）

01_extract_pairs.py との違い（DECISIONS.md D1–D7）:
  - Buckeye .words / .phones の時刻を「終了時刻」として読む（開始 = 直前エントリの終了）
  - .txt と .words の nth 対応を検査し、不一致事象を除外（D3）
  - 強度は前後 0.1 s の余白付きで抽出し語区間内で平均。peak-vowel intensity も測定（D5）
  - 局所調音速度（表層音節/秒、対象語を除く）（D6）
  - 事象レベル変数（cutoff 断片長、cutoff→repeat 間隔、retrace 長、介在語数）
  - 同話者・同語タイプの流暢ベースライン（D7）
  - データ選択の各段階の件数を selection_counts.csv に記録

出力（--out, 既定 ../output_v2/）:
  events.csv, tokens.csv（event + fluent）, selection_counts.csv

使い方:  python3 11_extract_v2.py
"""
from __future__ import annotations
import argparse
import glob
import importlib.util
import io
import os
import re
import zipfile

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("v1", os.path.join(HERE, "01_extract_pairs.py"))
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)  # scan_candidates / normalize_token / FILLERS を共有（事象定義は不変）

VOWEL_RE = re.compile(r"^(aa|ae|ah|ao|aw|ay|eh|er|ey|ih|iy|ow|oy|uh|uw|el|em|en|eng)n?$")
DISFL = ("<CUTOFF", "<HES", "<EXT", "<ERROR")
PAD = 0.1
MAX_PER_TYPE = 30
SEED = 1


# ── 読み込み ────────────────────────────────────────────────────────
def read_xlabel(path: str) -> pd.DataFrame:
    """Buckeye xlabel 形式（.words/.phones）。時刻は終了時刻、開始は直前エントリの終了。"""
    rows, prev = [], 0.0
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    i0 = next(i for i, l in enumerate(lines) if l.strip() == "#") + 1
    for l in lines[i0:]:
        p = l.split(None, 2)
        if len(p) < 3:
            continue
        try:
            end = float(p[0])
        except ValueError:
            continue
        fields = [x.strip() for x in p[2].split(";")]
        rows.append({"start": prev, "end": end, "label": fields[0],
                     "surface": fields[2] if len(fields) > 2 else ""})
        prev = end
    return pd.DataFrame(rows)


def load_words(path: str) -> pd.DataFrame:
    df = read_xlabel(path)
    df["norm"] = df["label"].map(v1.normalize_token)
    df["nsyl"] = df["surface"].map(lambda s: sum(bool(VOWEL_RE.match(ph)) for ph in s.split()))
    df["idx"] = np.arange(len(df))
    lex = df["norm"] != ""
    df.loc[lex, "nth"] = df[lex].groupby("norm").cumcount() + 1
    df["lex_i"] = np.nan
    df.loc[lex, "lex_i"] = np.arange(lex.sum())
    return df


def txt_line_final(txt_path: str) -> dict:
    """(word, nth) → .txt 行末なら 1（01_extract_pairs.py の is_final と同じ定義）。"""
    out, wc = {}, {}
    with open(txt_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            toks = line.split()
            valid = [i for i, t in enumerate(toks) if v1.normalize_token(t)]
            for i, t in enumerate(toks):
                n = v1.normalize_token(t)
                if n:
                    wc[n] = wc.get(n, 0) + 1
                    out[(n, wc[n])] = int(i == valid[-1])
    return out


def txt_between_lex(txt_path: str) -> dict:
    """事象ごとに .txt 上で cutoff と repeat の間にある語彙語数を (word, first_nth) で引けるようにする。"""
    toks, wc = [], {}
    with open(txt_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            for t in line.split():
                n = v1.normalize_token(t)
                nth = 0
                if n:
                    wc[n] = wc.get(n, 0) + 1
                    nth = wc[n]
                toks.append((t, n, nth))
    out = {}
    for i, (t, _, _) in enumerate(toks):
        if v1.CUTOFF_RE.match(t) and i > 0 and toks[i - 1][1]:
            w, nth = toks[i - 1][1], toks[i - 1][2]
            for j in range(i + 1, min(i + 6, len(toks))):
                if toks[j][1] == w:
                    out[(w, nth)] = sum(1 for x in toks[i + 1:j] if x[1])
                    break
    return out


# ── 測定 ────────────────────────────────────────────────────────────
def measure(snd, phones: pd.DataFrame, t0: float, t1: float) -> dict:
    res = {"intensity": np.nan, "int_peak_vowel": np.nan, "f0_k": 0, "f0_n": 0}
    try:
        a, b = max(0.0, t0 - PAD), min(snd.xmax, t1 + PAD)
        part = snd.extract_part(from_time=a, to_time=b, preserve_times=True)
        it = part.to_intensity(minimum_pitch=100)
        res["intensity"] = call(it, "Get mean", t0, t1, "energy")
        vow = phones[(phones.start >= t0 - 1e-4) & (phones.end <= t1 + 1e-4)
                     & phones.label.str.match(VOWEL_RE)]
        vals = [call(it, "Get mean", v.start, v.end, "energy") for v in vow.itertuples()
                if v.end - v.start >= 0.01]
        vals = [x for x in vals if x == x]
        if vals:
            res["int_peak_vowel"] = max(vals)
        pitch = part.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=300)
        ts = pitch.xs()
        f0 = pitch.selected_array["frequency"]
        inwin = (ts >= t0) & (ts <= t1)
        res["f0_n"] = int(inwin.sum())
        res["f0_k"] = int(((f0 > 75) & (f0 < 300) & inwin).sum())
    except Exception:  # noqa: BLE001
        pass
    return res


def local_rates(wdf: pd.DataFrame, row, win: float = 2.0) -> tuple[float, float]:
    mid = (row.start + row.end) / 2
    lex = wdf[(wdf.norm != "") & (wdf.end > mid - win) & (wdf.start < mid + win)
              & (wdf.idx != row.idx)]
    dur = (lex.end - lex.start).sum()
    return (lex.nsyl.sum() / dur if dur > 0 else np.nan, len(lex) / (2 * win))


def prev_pause(wdf: pd.DataFrame, i: int) -> int:
    """直前が 100 ms 以上の <SIL>、あるいは <IVER>/{B_TRANS}（話者交替・冒頭）なら 1。"""
    if i == 0:
        return 1
    p = wdf.iloc[i - 1]
    return int((p.label == "<SIL>" and p.end - p.start >= 0.1)
               or p.label.startswith(("<IVER", "{B_TRANS")))


def token_record(wdf, phones, snd, row, **extra) -> dict:
    rate_syl, rate_wps = local_rates(wdf, row)
    m = measure(snd, phones, row.start, row.end)
    return {"start": row.start, "end": row.end, "duration": row.end - row.start,
            "nsyl": row.nsyl, "rate_syl": rate_syl, "rate_wps": rate_wps,
            "prev_pause": prev_pause(wdf, int(row.idx)), **m, **extra}


def retrace_length(wdf: pd.DataFrame, f, r) -> int:
    lex = wdf[wdf.norm != ""].reset_index(drop=True)
    pf, pr = int(f.lex_i), int(r.lex_i)
    k = 0
    while pf - k >= 0 and pr - k > pf and lex.norm[pf - k] == lex.norm[pr - k]:
        k += 1
    return k


# ── 頻度 ────────────────────────────────────────────────────────────
def load_frequency(cache: str) -> dict:
    if os.path.exists(cache):
        d = pd.read_csv(cache, sep="\t", keep_default_na=False)
    else:
        import requests
        r = requests.get(v1.SUBTLEX_URL, timeout=60)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        name = next(f for f in z.namelist() if "SUBTLEX" in f and f.endswith(".txt"))
        with z.open(name) as f:
            d = pd.read_csv(f, sep="\t", keep_default_na=False)
        d[["Word", "Lg10WF"]].to_csv(cache, sep="\t", index=False)
    return dict(zip(d["Word"].astype(str).str.lower(), d["Lg10WF"]))


# ── メイン ──────────────────────────────────────────────────────────
def main() -> None:
    root = os.path.dirname(HERE)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=os.path.join(root, "raw"))
    ap.add_argument("--out", default=os.path.join(root, "output_v2"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    freq = load_frequency(os.path.join(root, "intermediate", "subtlexus_lg10wf.tsv"))
    cand = v1.scan_candidates(args.raw)
    cand["event_id"] = np.arange(len(cand))
    counts = {"cutoff_markers": sum(len(v1.CUTOFF_RE.findall(open(p, errors="ignore").read()))
                                    for p in glob.glob(os.path.join(args.raw, "*.txt"))),
              "candidate_events": len(cand)}

    events, tokens = [], []
    for spk, sg in cand.groupby("speaker"):
        event_types = set(sg.word)
        fluent_pool = []
        for txt in sorted(glob.glob(os.path.join(args.raw, f"{spk}*.txt"))):
            base = os.path.splitext(os.path.basename(txt))[0]
            wpath = os.path.join(args.raw, f"{base}.words")
            if not os.path.exists(wpath):
                continue
            wdf = load_words(wpath)
            phones = read_xlabel(os.path.join(args.raw, f"{base}.phones"))
            snd = parselmouth.Sound(os.path.join(args.raw, f"{base}.wav"))
            final = txt_line_final(txt)
            tbetween = txt_between_lex(txt)
            used = set()
            for r in sg[sg.basename == base].itertuples():
                ev = {"event_id": r.event_id, "basename": base, "speaker": spk, "word": r.word}
                if r.word in v1.FILLERS:
                    events.append({**ev, "status": "filler_target"}); continue
                fr = wdf[(wdf.norm == r.word) & (wdf.nth == r.first_nth)]
                rr = wdf[(wdf.norm == r.word) & (wdf.nth == r.repeat_nth)]
                if fr.empty or rr.empty:
                    events.append({**ev, "status": "not_in_words"}); continue
                f, rp = fr.iloc[0], rr.iloc[0]
                nxt = wdf.iloc[int(f.idx) + 1:int(rp.idx)]
                nxt_nonsil = nxt[nxt.label != "<SIL>"]
                between = nxt
                cut = between[between.label.str.startswith("<CUTOFF")]
                n_between = int((between.norm != "").sum())
                ok = (not nxt_nonsil.empty and nxt_nonsil.iloc[0].label.startswith("<CUTOFF")
                      and n_between == tbetween.get((r.word, r.first_nth), -1))
                if not ok:
                    events.append({**ev, "status": "misaligned"}); continue
                c_last = cut.iloc[-1]
                after = between[between.idx > c_last.idx]
                used.update([int(f.idx), int(rp.idx)])
                events.append({**ev, "status": "ok", "n_between_lex": n_between,
                               "cut_frag_dur": float((cut.end - cut.start).sum()),
                               "cut_label": cut.iloc[0].label,
                               "gap_cut_to_repeat": float(rp.start - c_last.end),
                               "sil_after_cut": float((after.end - after.start)[after.label == "<SIL>"].sum()),
                               "retrace_len": retrace_length(wdf, f, rp)})
                for cond, w, nth in (("first", f, r.first_nth), ("repeat", rp, r.repeat_nth)):
                    tokens.append(token_record(wdf, phones, snd, w, event_id=r.event_id,
                                               basename=base, speaker=spk, word=r.word,
                                               condition=cond,
                                               utt_pos=final.get((r.word, nth), 0)))
            # 流暢ベースライン候補（D7）
            lexmask = wdf.norm != ""
            near = set()
            for i in wdf.index[wdf.label.str.startswith(DISFL)]:
                near.update(range(i - 2, i + 3))
            lex = wdf[lexmask]
            prev_n, next_n = lex.norm.shift(1), lex.norm.shift(-1)
            bad_nb = (prev_n.isin(v1.FILLERS) | next_n.isin(v1.FILLERS)
                      | (prev_n == lex.norm) | (next_n == lex.norm))
            pool = lex[lex.norm.isin(event_types) & ~bad_nb & ~lex.idx.isin(near | used)]
            for w in pool.itertuples():
                fluent_pool.append((base, w.idx, w.norm, final.get((w.norm, int(w.nth)), 0)))
        # 話者×語タイプごとに最大 MAX_PER_TYPE をサンプリングして測定
        fp = pd.DataFrame(fluent_pool, columns=["basename", "idx", "word", "utt_pos"])
        if fp.empty:
            continue
        sel = fp.sample(frac=1, random_state=SEED).groupby("word").head(MAX_PER_TYPE)
        for base, bg in sel.groupby("basename"):
            wdf = load_words(os.path.join(args.raw, f"{base}.words"))
            phones = read_xlabel(os.path.join(args.raw, f"{base}.phones"))
            snd = parselmouth.Sound(os.path.join(args.raw, f"{base}.wav"))
            for s in bg.itertuples():
                tokens.append(token_record(wdf, phones, snd, wdf.iloc[int(s.idx)], event_id=-1,
                                           basename=base, speaker=spk, word=s.word,
                                           condition="fluent", utt_pos=s.utt_pos))
        print(f"  {spk}: events={len(sg)} fluent={len(sel)}")

    ev = pd.DataFrame(events)
    tok = pd.DataFrame(tokens)
    # 主: Buckeye 全体での語頻度（log10 per million）。縮約形も同じ基準で数えられる（D10）
    allw = pd.concat([load_words(p).norm for p in glob.glob(os.path.join(args.raw, "*.words"))])
    allw = allw[allw != ""]
    bfreq = np.log10(allw.value_counts() / len(allw) * 1e6)
    tok["log_freq"] = tok.word.map(bfreq)
    # 副: SUBTLEX-US Lg10WF（縮約形は分割表記のため欠損 → 最小値を付与）
    fvals = pd.Series({w: freq.get(w, np.nan) for w in tok.word.unique()})
    tok["log_freq_subtlex"] = tok.word.map(fvals.fillna(fvals.min()))
    tok["freq_missing"] = tok.word.map(fvals.isna()).astype(int)
    tok["keep_dur"] = ((tok.duration >= 0.05) & (tok.duration <= 2.0)).astype(int)
    tok["keep_int"] = (tok.keep_dur.astype(bool) & (tok.intensity >= 20)).astype(int)

    okev = ev[ev.status == "ok"].event_id
    et = tok[tok.event_id >= 0]
    kd = et[et.keep_dur == 1]
    ki = et[et.keep_int == 1]
    counts.update({
        "events_not_in_words": int((ev.status == "not_in_words").sum()),
        "events_filler_target": int((ev.status == "filler_target").sum()),
        "events_misaligned": int((ev.status == "misaligned").sum()),
        "events_aligned": len(okev),
        "tokens_first_dur": int((kd.condition == "first").sum()),
        "tokens_repeat_dur": int((kd.condition == "repeat").sum()),
        "excluded_first_dur": int(((et.keep_dur == 0) & (et.condition == "first")).sum()),
        "excluded_repeat_dur": int(((et.keep_dur == 0) & (et.condition == "repeat")).sum()),
        "complete_pairs_dur": int((kd.groupby("event_id").size() == 2).sum()),
        "events_any_token_dur": int(kd.event_id.nunique()),
        "tokens_first_int": int((ki.condition == "first").sum()),
        "tokens_repeat_int": int((ki.condition == "repeat").sum()),
        "complete_pairs_int": int((ki.groupby("event_id").size() == 2).sum()),
        "fluent_tokens_dur": int(((tok.event_id < 0) & (tok.keep_dur == 1)).sum()),
        "fluent_tokens_int": int(((tok.event_id < 0) & (tok.keep_int == 1)).sum()),
        "speakers": int(kd.speaker.nunique()),
        "word_types": int(kd.word.nunique()),
        "freq_missing_types": int(fvals.isna().sum()),
    })
    ev.to_csv(os.path.join(args.out, "events.csv"), index=False)
    tok.to_csv(os.path.join(args.out, "tokens.csv"), index=False)
    pd.Series(counts).to_csv(os.path.join(args.out, "selection_counts.csv"), header=["n"])
    print(pd.Series(counts).to_string())


if __name__ == "__main__":
    main()
