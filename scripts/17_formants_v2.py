#!/usr/bin/env python3
"""
17_formants_v2.py — 母音の中央化（フォルマント）による縮約の指標（DECISIONS.md D23）

持続時間は語頭境界の誤りに左右される（D19–D21）。母音の中点のフォルマントは境界が数十 ms ずれても
ほとんど変わらないので、縮約の別の指標として測る。反復縮約は母音の中央化も伴うことが多い。

手順:
  - tokens.csv の各トークン（keep_dur == 1）について、語区間に中点が入る母音（.phones）のうち最も長いものを取る。
  - 母音の中点で F1・F2 を測る（Praat Burg、窓 25 ms、5 本、上限は女性 5500 Hz・男性 5000 Hz。
    話者の性別は Buckeye Corpus manual の話者表による）。母音が 30 ms 未満のトークンは測らない。
  - 話者ごとに Lobanov 正規化（話者内の全測定母音で z 化）し、話者×母音ラベルごとに |z| > 3 の値を外れ値として除く。
  - 中央化の指標: 話者の母音空間の中心（正規化値の平均）からの距離。小さいほど中央化（縮約）している。

入力:  output_v2/tokens.csv、raw/*.phones・*.wav
出力:  output_v2/formants_v2.csv（tokens.csv の行番号 tok_row と測定値。音素ラベルは母音の種類のみ、時刻なし）
使い方:  python3 scripts/17_formants_v2.py
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output_v2")
RAW = os.path.join(ROOT, "raw")
_spec = importlib.util.spec_from_file_location("x11", os.path.join(HERE, "11_extract_v2.py"))
x11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(x11)  # read_xlabel・VOWEL_RE を共有

# Buckeye Corpus manual の話者表（話者の性別）
FEMALE = {"s01", "s02", "s04", "s05", "s07", "s08", "s09", "s12", "s14", "s16", "s17", "s18", "s20",
          "s21", "s25", "s26", "s27", "s31", "s37", "s39"}
MIN_VOWEL = 0.03
PAD = 0.05


def main() -> None:
    tok = pd.read_csv(os.path.join(OUT, "tokens.csv"))
    tok["tok_row"] = np.arange(len(tok))
    tok = tok[tok.keep_dur == 1]
    rows = []
    for base, g in tok.groupby("basename"):
        ph = x11.read_xlabel(os.path.join(RAW, f"{base}.phones"))
        vow = ph[ph.label.str.match(x11.VOWEL_RE)].copy()
        vow["mid"] = (vow.start + vow.end) / 2
        snd = parselmouth.Sound(os.path.join(RAW, f"{base}.wav"))
        ceiling = 5500 if g.speaker.iloc[0] in FEMALE else 5000
        for r in g.itertuples():
            v = vow[(vow.mid > r.start) & (vow.mid < r.end)]
            rec = {"tok_row": r.tok_row, "vowel": "", "vdur": np.nan, "F1": np.nan, "F2": np.nan}
            if len(v):
                v = v.loc[(v.end - v.start).idxmax()]
                rec["vowel"] = v.label.rstrip("n") if v.label not in ("en", "em", "eng") else v.label
                rec["vdur"] = v.end - v.start
                if rec["vdur"] >= MIN_VOWEL:
                    try:
                        part = snd.extract_part(from_time=max(0, v.start - PAD), to_time=min(snd.xmax, v.end + PAD),
                                                preserve_times=True)
                        fm = part.to_formant_burg(time_step=0.005, max_number_of_formants=5,
                                                  maximum_formant=ceiling, window_length=0.025)
                        t = (v.start + v.end) / 2
                        rec["F1"] = call(fm, "Get value at time", 1, t, "hertz", "Linear")
                        rec["F2"] = call(fm, "Get value at time", 2, t, "hertz", "Linear")
                    except Exception:  # noqa: BLE001
                        pass
            rows.append(rec)
    f = pd.DataFrame(rows)
    f = f.merge(tok[["tok_row", "speaker"]], on="tok_row")
    ok = f.F1.notna() & f.F2.notna()
    for k in ("F1", "F2"):
        grp = f[ok].groupby("speaker")[k]
        f.loc[ok, f"z{k}"] = (f.loc[ok, k] - grp.transform("mean")) / grp.transform("std")
    # 話者×母音ごとの外れ値（追跡の誤り）を除く
    for k in ("zF1", "zF2"):
        g2 = f[ok].groupby(["speaker", "vowel"])[k]
        dev = (f.loc[ok, k] - g2.transform("mean")) / g2.transform("std")
        f.loc[ok & (dev.abs() > 3).reindex(f.index, fill_value=False), ["zF1", "zF2"]] = np.nan
    f["central_dist"] = np.sqrt(f.zF1 ** 2 + f.zF2 ** 2)
    f.drop(columns="speaker").to_csv(os.path.join(OUT, "formants_v2.csv"), index=False)
    print(f"tokens {len(f)}, with vowel {(f.vowel != '').sum()}, measured {f.central_dist.notna().sum()}")


if __name__ == "__main__":
    main()
