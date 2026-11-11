#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14_boundary_check_sample.py — 語区間の目視確認キット用サンプリングとクリップ切り出し（R1 §2.2）

目的: 修正版抽出（11_extract_v2.py）で使った語区間（開始 = 直前エントリの終了）が
      実際の音声と合っているかを、著者が約 50 トークンを聴いて確認するための素材を作る。

方針:
  - keep_dur==1 のトークンから、反復事象 20 件（first・repeat の両方が keep_dur==1）= 40 トークン
    + 流暢トークン 10 件 = 50 トークンを、固定シードで抽出する。
  - 話者ごとに最大 1 件（事象 20 話者 + 流暢 10 話者、事象と流暢の話者も重ならないよう優先）。
  - 提示順をシャッフルし、items.csv には条件（first/repeat/fluent）を載せない（ブラインド判定）。
    条件は key.csv にのみ記録する。
  - 各トークンについて、tokens.csv の (start, end) が .words を正しい読み方
    （時刻 = 終了時刻、開始 = 直前エントリの終了）で読んだ区間と一致することを検査する。
  - クリップは語区間の前後 0.5 s（ファイル範囲でクランプ）。TextGrid は
    "target"（分析区間）と "words"（クリップ内の Buckeye 語）の 2 層。

入力（--tokens, --raw）:
  output_v2/tokens.csv, raw/<basename>.wav, raw/<basename>.words
出力（--out, 既定 ../boundary_check/。.gitignore 対象。Buckeye の音声・語ラベル・時刻を含むので外部に出さない）:
  clips/<item_id>.wav, clips/<item_id>.TextGrid, items.csv, key.csv, sample_info.txt
  既存の出力があれば上書きせず中止する（--force で上書き）。

使い方:  python3 scripts/14_boundary_check_sample.py
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
# 11_extract_v2.py の read_xlabel（終了時刻として読む正しい reader）をそのまま使う
_spec = importlib.util.spec_from_file_location("v2", os.path.join(HERE, "11_extract_v2.py"))
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

SEED = 20241105
N_EVENTS = 20
N_FLUENT = 10
PAD = 0.5      # 語区間の前後に付ける余白（秒）
TOL = 1e-6     # tokens.csv と .words 由来区間の許容差（秒）


def tg_escape(s: str) -> str:
    return str(s).replace('"', '""')


def write_textgrid(path: str, xmax: float, tiers: list[tuple[str, list[tuple[float, float, str]]]]) -> None:
    """Praat の long text 形式 TextGrid。各 tier は [0, xmax] を隙間なく覆う区間のリスト。"""
    out = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
           "xmin = 0", f"xmax = {xmax:.6f}", "tiers? <exists>", f"size = {len(tiers)}", "item []:"]
    for k, (name, ivs) in enumerate(tiers, 1):
        out += [f"    item [{k}]:", '        class = "IntervalTier"', f'        name = "{tg_escape(name)}"',
                "        xmin = 0", f"        xmax = {xmax:.6f}", f"        intervals: size = {len(ivs)}"]
        for j, (a, b, lab) in enumerate(ivs, 1):
            out += [f"        intervals [{j}]:", f"            xmin = {a:.6f}", f"            xmax = {b:.6f}",
                    f'            text = "{tg_escape(lab)}"']
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def find_interval(wdf: pd.DataFrame, start: float, end: float, word: str):
    """tokens.csv の (start, end, word) と一致する .words のエントリを返す。なければ None。"""
    hit = wdf[(wdf.start.sub(start).abs() < TOL) & (wdf.end.sub(end).abs() < TOL)]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return r if r.norm == word else None


def main() -> None:
    root = os.path.dirname(HERE)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", default=os.path.join(root, "output_v2", "tokens.csv"))
    ap.add_argument("--raw", default=os.path.join(root, "raw"))
    ap.add_argument("--out", default=os.path.join(root, "boundary_check"))
    ap.add_argument("--force", action="store_true", help="既存の出力を上書きする")
    args = ap.parse_args()

    if os.path.exists(os.path.join(args.out, "items.csv")) and not args.force:
        sys.exit(f"既存の出力があります: {args.out}（上書きするなら --force）")
    clips_dir = os.path.join(args.out, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    tok = pd.read_csv(args.tokens)
    tok = tok[(tok.keep_dur == 1)
              & tok.basename.map(lambda b: os.path.exists(os.path.join(args.raw, f"{b}.wav"))
                                 and os.path.exists(os.path.join(args.raw, f"{b}.words")))]
    rng = np.random.default_rng(SEED)

    # 反復事象: first と repeat の両方が keep_dur==1
    ev = tok[tok.event_id >= 0]
    ok_ev = ev.groupby("event_id").filter(lambda g: len(g) == 2 and set(g.condition) == {"first", "repeat"})
    ev_by_spk = {s: sorted(g.event_id.unique()) for s, g in ok_ev.groupby("speaker")}
    spk_ev = sorted(ev_by_spk)
    pick_spk = list(rng.choice(spk_ev, size=N_EVENTS, replace=False))
    pick_events = [int(rng.choice(ev_by_spk[s])) for s in pick_spk]

    # 流暢: 事象に使わなかった話者を優先して 1 話者 1 トークン
    fl = tok[tok.condition == "fluent"]
    fl_spk = sorted(set(fl.speaker) - set(pick_spk)) or sorted(set(fl.speaker))
    if len(fl_spk) < N_FLUENT:
        fl_spk = sorted(set(fl.speaker))
    fl_pick_spk = list(rng.choice(fl_spk, size=N_FLUENT, replace=False))
    fl_rows = [fl[fl.speaker == s].iloc[int(rng.integers(len(fl[fl.speaker == s])))] for s in fl_pick_spk]

    sel = pd.concat([ok_ev[ok_ev.event_id.isin(pick_events)], pd.DataFrame(fl_rows)])
    sel = sel.sample(frac=1, random_state=SEED).reset_index(drop=True)  # 提示順シャッフル
    sel["item_id"] = [f"bc{i + 1:02d}" for i in range(len(sel))]

    items, keys, mismatches = [], [], []
    cache: dict[str, tuple] = {}
    for r in sel.itertuples():
        if r.basename not in cache:
            cache[r.basename] = (v2.load_words(os.path.join(args.raw, f"{r.basename}.words")),
                                 parselmouth.Sound(os.path.join(args.raw, f"{r.basename}.wav")))
        wdf, snd = cache[r.basename]
        hit = find_interval(wdf, r.start, r.end, r.word)
        if hit is None:
            mismatches.append(r.item_id)
            continue  # 一致しないトークンはクリップを作らず、件数として報告する
        c0, c1 = max(0.0, r.start - PAD), min(snd.xmax, r.end + PAD)
        snd.extract_part(from_time=c0, to_time=c1, preserve_times=False).save(
            os.path.join(clips_dir, f"{r.item_id}.wav"), "WAV")
        dur = c1 - c0
        t0, t1 = r.start - c0, r.end - c0
        # words 層: クリップに重なる Buckeye エントリをクリップ時刻に直し、端を切る
        w = wdf[(wdf.end > c0 + 1e-9) & (wdf.start < c1 - 1e-9)]
        w_ivs = [(max(x.start, c0) - c0, min(x.end, c1) - c0, x.label) for x in w.itertuples()]
        w_ivs[0] = (0.0, w_ivs[0][1], w_ivs[0][2])
        w_ivs[-1] = (w_ivs[-1][0], dur, w_ivs[-1][2])
        t_ivs = [(0.0, t0, ""), (t0, t1, r.word), (t1, dur, "")]
        write_textgrid(os.path.join(clips_dir, f"{r.item_id}.TextGrid"), dur,
                       [("target", t_ivs), ("words", w_ivs)])
        # 前後の語彙語（<SIL> や <CUTOFF> などのタグは飛ばす）
        i = int(hit.idx)
        prev_w = wdf[(wdf.idx < i) & (wdf.norm != "")].tail(1)
        next_w = wdf[(wdf.idx > i) & (wdf.norm != "")].head(1)
        items.append({"item_id": r.item_id, "clip_file": f"clips/{r.item_id}.wav", "word": r.word,
                      "target_start_in_clip": round(t0, 6), "target_end_in_clip": round(t1, 6),
                      "prev_word": prev_w.label.iloc[0] if len(prev_w) else "",
                      "next_word": next_w.label.iloc[0] if len(next_w) else ""})
        keys.append({"item_id": r.item_id, "event_id": r.event_id, "basename": r.basename,
                     "speaker": r.speaker, "condition": r.condition,
                     "orig_start": r.start, "orig_end": r.end})

    pd.DataFrame(items).to_csv(os.path.join(args.out, "items.csv"), index=False)
    pd.DataFrame(keys).to_csv(os.path.join(args.out, "key.csv"), index=False)
    kd = pd.DataFrame(keys)
    cnt = kd.condition.value_counts().to_dict()
    info = [f"seed = {SEED}", f"pad = {PAD} s", f"tokens = {len(sel)}",
            f"first = {cnt.get('first', 0)}, repeat = {cnt.get('repeat', 0)}, fluent = {cnt.get('fluent', 0)}",
            f"speakers = {kd.speaker.nunique()}",
            f"interval_mismatches (tokens.csv vs .words) = {len(mismatches)} of {len(sel)}"
            + (f" [{', '.join(mismatches)}]" if mismatches else "")]
    with open(os.path.join(args.out, "sample_info.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(info) + "\n")
    print("\n".join(info))


if __name__ == "__main__":
    main()
