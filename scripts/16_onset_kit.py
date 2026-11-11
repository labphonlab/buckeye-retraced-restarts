#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16_onset_kit.py — 言い直しの語頭を人手で付け直すためのキット（DECISIONS.md D23）

目的: R が cutoff 断片の直後に始まる事象（gap_cut_to_repeat == 0）では、コーパスの R の語頭境界が
      断片の末尾を含んでいる疑いがある（D19–D21）。著者が音声とスペクトログラムを見て語頭を付け直し、
      その位置で持続時間・強度を測り直す。

項目:
  - 直後型の R すべて（keep_dur == 1）
  - 対照: 間のある事象の R 32 件、F 32 件（固定シード）。付け直しの系統的なずれが直後型 R に
    特有かを確かめ、判定者とコーパスの境界の通常の差を見積もるため。
  提示順はシャッフルし、items.csv には条件も、コーパスの境界の位置も載せない（ブラインド）。
  クリップの切り出し位置は項目ごとに乱数でずらし、境界がクリップ内の決まった位置に来ないようにする。

出力（--out、既定 ../onset_check/。.gitignore 対象。Buckeye の音声・時刻を含むので外部に出さない）:
  clips/<item_id>.wav, items.csv（item_id, clip_file, word, before, after）,
  key.csv（条件・コーパス時刻・クリップ開始時刻）, kit_info.txt
  既存の出力があれば上書きせず中止する（--force で上書き）。

使い方:  python3 scripts/16_onset_kit.py
"""
from __future__ import annotations
import argparse
import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd
import parselmouth

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("v2", os.path.join(HERE, "11_extract_v2.py"))
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)  # read_xlabel / load_words（時刻 = 終了時刻）

SEED = 20261105
N_CTRL = 32            # 対照（間のある R、F）各群の件数
PRE = (0.35, 0.65)     # 語頭の前に付ける余白（秒、一様乱数）
POST = (0.15, 0.35)    # 語末の後に付ける余白（秒、一様乱数）
TOL = 1e-6
CUT_RE = re.compile(r"^<CUTOFF-([^=>]*)")


def show_label(lab: str, norm: str) -> str:
    """文脈表示用。語はそのまま、断片は 'ju-'、無音は '…'、その他のタグは出さない。"""
    if norm:
        return lab
    m = CUT_RE.match(lab)
    if m:
        return m.group(1) + "-"
    if lab.startswith("<SIL"):
        return "…"
    return ""


def context(wdf: pd.DataFrame, i: int, step: int, n: int) -> str:
    out, j = [], i + step
    while 0 <= j < len(wdf) and len(out) < n:
        s = show_label(wdf.label.iloc[j], wdf.norm.iloc[j])
        if s:
            out.append(s)
        j += step
    return " ".join(out[::-1] if step < 0 else out)


def main() -> None:
    root = os.path.dirname(HERE)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(root, "onset_check"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.exists(os.path.join(args.out, "items.csv")) and not args.force:
        sys.exit(f"既存の出力があります: {args.out}（上書きするなら --force）")
    clips = os.path.join(args.out, "clips")
    os.makedirs(clips, exist_ok=True)
    raw = os.path.join(root, "raw")

    tok = pd.read_csv(os.path.join(root, "output_v2", "tokens.csv"))
    ev = pd.read_csv(os.path.join(root, "output_v2", "events.csv"))
    ev = ev[ev.status == "ok"]
    abut = set(ev.event_id[ev.gap_cut_to_repeat == 0])
    et = tok[(tok.condition != "fluent") & (tok.keep_dur == 1)].copy()
    et["group"] = np.where(et.condition == "first", "first",
                           np.where(et.event_id.isin(abut), "repeat_abut", "repeat_gap"))
    rng = np.random.default_rng(SEED)
    sel = pd.concat([
        et[et.group == "repeat_abut"],
        et[et.group == "repeat_gap"].sample(N_CTRL, random_state=SEED),
        et[et.group == "first"].sample(N_CTRL, random_state=SEED + 1),
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    sel["item_id"] = [f"on{i + 1:03d}" for i in range(len(sel))]

    items, keys, mism, cache = [], [], [], {}
    for r in sel.itertuples():
        if r.basename not in cache:
            cache[r.basename] = (v2.load_words(os.path.join(raw, f"{r.basename}.words")),
                                 parselmouth.Sound(os.path.join(raw, f"{r.basename}.wav")))
        wdf, snd = cache[r.basename]
        hit = wdf[(wdf.start.sub(r.start).abs() < TOL) & (wdf.end.sub(r.end).abs() < TOL)]
        if hit.empty or hit.norm.iloc[0] != r.word:
            mism.append(r.item_id)
            continue
        i = int(hit.idx.iloc[0])
        c0 = max(0.0, r.start - rng.uniform(*PRE))
        c1 = min(snd.xmax, r.end + rng.uniform(*POST))
        snd.extract_part(from_time=c0, to_time=c1, preserve_times=False).save(
            os.path.join(clips, f"{r.item_id}.wav"), "WAV")
        items.append({"item_id": r.item_id, "clip_file": f"clips/{r.item_id}.wav", "word": r.word,
                      "before": context(wdf, i, -1, 3), "after": context(wdf, i, 1, 2)})
        keys.append({"item_id": r.item_id, "event_id": r.event_id, "basename": r.basename,
                     "speaker": r.speaker, "condition": r.condition, "group": r.group,
                     "orig_start": r.start, "orig_end": r.end, "clip_start": c0, "clip_end": c1})

    pd.DataFrame(items).to_csv(os.path.join(args.out, "items.csv"), index=False)
    kd = pd.DataFrame(keys)
    kd.to_csv(os.path.join(args.out, "key.csv"), index=False)
    info = [f"seed = {SEED}", f"items = {len(kd)}",
            "groups = " + ", ".join(f"{k} {v}" for k, v in kd.group.value_counts().sort_index().items()),
            f"speakers = {kd.speaker.nunique()}",
            f"interval_mismatches (tokens.csv vs .words) = {len(mism)}" + (f" [{', '.join(mism)}]" if mism else "")]
    with open(os.path.join(args.out, "kit_info.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(info) + "\n")
    print("\n".join(info))


if __name__ == "__main__":
    main()
