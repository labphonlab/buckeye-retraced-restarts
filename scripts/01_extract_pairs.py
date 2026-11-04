#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_extract_pairs.py — 同語反復（same-word restart）ペアの抽出と音響測定

Google Colab notebook `re-encoding_after_self-repair.ipynb`（Cell 1）のローカル移植版。
Colab 依存（drive.mount / rpy2 の %%R マジック）を除去し、ローカルの raw/ から実行する。

処理:
  1. SUBTLEX-US から語頻度（Lg10WF）を取得（失敗時はダミー0）
  2. raw/*.txt（Buckeye 転記）を走査し、<CUTOFF-...> マーカー直前の語を
     直後5語以内で言い直している「同語反復ペア（first / repeat）」を検出
  3. 各ペアについて raw/*.words から onset/offset・duration、raw/*.wav から
     parselmouth で intensity・F0可用性（信頼フレーム数/総フレーム数）を測定、
     局所話速（±2秒窓の語数/秒）も算出
  4. data_for_r.csv（分析用）・Table1_Descriptive_Stats.csv・Figure1/2/S1 を出力

使い方（既定でリポジトリ構成を自動解決）:
    python3 01_extract_pairs.py
    python3 01_extract_pairs.py --raw ../raw --out ../output --intermediate ../intermediate
"""
from __future__ import annotations
import argparse
import glob
import io
import os
import re
import string
import zipfile

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

SUBTLEX_URL = ("https://www.ugent.be/pp/experimentele-psychologie/en/research/"
               "documents/subtlexus/subtlexus2.zip")
FILLERS = {"uh", "um", "mm", "hm", "ah", "eh"}
CUTOFF_RE = re.compile(r"<CUTOFF-[^>]+>")


# ── ヘルパー ────────────────────────────────────────────────────────
def normalize_token(token: str) -> str:
    if not token:
        return ""
    if token.startswith("<") or token.startswith("{"):
        return ""
    return token.strip(string.punctuation).lower()


def load_words_robust(filepath: str) -> pd.DataFrame:
    """Buckeye .words ローダ: 次語の onset を当該語の offset として duration を算出。"""
    if not os.path.exists(filepath):
        return pd.DataFrame()
    data = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and not line.startswith(("#", "{")):
                try:
                    onset = float(parts[0])
                    raw_word = parts[2]
                    norm_w = normalize_token(raw_word)
                    if norm_w:
                        data.append({"onset": onset, "word": norm_w})
                except ValueError:
                    continue
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["offset"] = df["onset"].shift(-1)
    df = df.dropna(subset=["offset"])
    df["duration"] = df["offset"] - df["onset"]
    df = df[df["duration"] > 0]
    df["nth"] = df.groupby("word").cumcount() + 1
    return df


def get_acoustic_measures(wav_path: str, t_min: float, t_max: float):
    """(Intensity_mean, F0_reliable_frames, F0_total_frames) を返す。"""
    try:
        if not os.path.exists(wav_path):
            return np.nan, 0, 0
        sound = parselmouth.Sound(wav_path)
        part = sound.extract_part(from_time=t_min, to_time=t_max, preserve_times=True)
        intensity = call(part.to_intensity(), "Get mean", t_min, t_max, "energy")
        pitch = part.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=300)
        f0_vals = pitch.selected_array["frequency"]
        n_total = len(f0_vals)
        n_reliable = int(np.sum((f0_vals > 75) & (f0_vals < 300)))
        return intensity, n_reliable, n_total
    except Exception:
        return np.nan, 0, 0


def calculate_local_speech_rate(wdf: pd.DataFrame, target_onset: float,
                                window: float = 2.0) -> float:
    start, end = target_onset - window, target_onset + window
    count = wdf[(wdf["onset"] >= start) & (wdf["onset"] <= end)].shape[0]
    return count / (2 * window)


# ── 頻度取得 ────────────────────────────────────────────────────────
def load_frequency() -> dict:
    print("Phase 0: Frequency data (SUBTLEX-US)...")
    try:
        import requests
        r = requests.get(SUBTLEX_URL, timeout=60)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        name = next(f for f in z.namelist() if "SUBTLEX" in f and f.endswith(".txt"))
        with z.open(name) as f:
            df_freq = pd.read_csv(f, sep="\t")
        return (df_freq.set_index(df_freq["Word"].astype(str).str.lower())["Lg10WF"]
                .to_dict())
    except Exception as e:  # noqa: BLE001
        print(f"  ! Frequency download failed ({e}); using dummy 0.0.")
        return {}


# ── ペア検出 ────────────────────────────────────────────────────────
def scan_candidates(raw_dir: str) -> pd.DataFrame:
    print("Phase 1: Corpus scanning & pairing...")
    candidates = []
    txt_files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    for txt_path in txt_files:
        base = os.path.splitext(os.path.basename(txt_path))[0]
        speaker = base[:3]
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        tokens = []
        wc: dict = {}
        for line in lines:
            l_toks = line.split()
            valid_idxs = [i for i, t in enumerate(l_toks) if normalize_token(t)]
            for idx, t in enumerate(l_toks):
                norm = normalize_token(t)
                nth, is_final = 0, 0
                if norm:
                    wc[norm] = wc.get(norm, 0) + 1
                    nth = wc[norm]
                    if valid_idxs and idx == valid_idxs[-1]:
                        is_final = 1
                tokens.append({"orig": t, "norm": norm, "nth": nth,
                               "is_final": is_final})

        for i, tok in enumerate(tokens):
            if CUTOFF_RE.match(tok["orig"]) and i > 0:
                target = tokens[i - 1]["norm"]
                if not target:
                    continue
                for j in range(i + 1, min(i + 6, len(tokens))):
                    mid = tokens[i + 1:j]
                    if any(m["orig"].startswith("<HES") or m["norm"] in FILLERS
                           for m in mid):
                        break
                    if tokens[j]["norm"] == target:
                        candidates.append({
                            "basename": base, "speaker": speaker, "word": target,
                            "first_nth": tokens[i - 1]["nth"],
                            "repeat_nth": tokens[j]["nth"],
                            "first_pos": tokens[i - 1]["is_final"],
                            "repeat_pos": tokens[j]["is_final"],
                        })
                        break
    return pd.DataFrame(candidates)


# ── ペアリング＋音響 ────────────────────────────────────────────────
def build_pairs(cand_df: pd.DataFrame, raw_dir: str, freq_dict: dict) -> pd.DataFrame:
    print("Phase 2: Pairing & acoustics...")
    final_pairs = []
    if cand_df.empty:
        return pd.DataFrame()
    for base, group in cand_df.groupby("basename"):
        wdf = load_words_robust(os.path.join(raw_dir, f"{base}.words"))
        if wdf.empty:
            continue
        wav_path = os.path.join(raw_dir, f"{base}.wav")
        for _, row in group.iterrows():
            f_rows = wdf[(wdf["word"] == row["word"]) & (wdf["nth"] == row["first_nth"])]
            r_rows = wdf[(wdf["word"] == row["word"]) & (wdf["nth"] == row["repeat_nth"])]
            if f_rows.empty or r_rows.empty:
                continue
            f, r = f_rows.iloc[0], r_rows.iloc[0]
            int_f, f0_k_f, f0_n_f = get_acoustic_measures(wav_path, f["onset"], f["offset"])
            int_r, f0_k_r, f0_n_r = get_acoustic_measures(wav_path, r["onset"], r["offset"])
            common = {"basename": base, "speaker": row["speaker"], "word": row["word"],
                      "log_freq": freq_dict.get(row["word"], 0.0)}
            final_pairs.append({**common, "condition": "first", "duration": f["duration"],
                                "intensity": int_f, "f0_k": f0_k_f, "f0_n": f0_n_f,
                                "utt_pos": row["first_pos"],
                                "speech_rate": calculate_local_speech_rate(wdf, f["onset"])})
            final_pairs.append({**common, "condition": "repeat", "duration": r["duration"],
                                "intensity": int_r, "f0_k": f0_k_r, "f0_n": f0_n_r,
                                "utt_pos": row["repeat_pos"],
                                "speech_rate": calculate_local_speech_rate(wdf, r["onset"])})
    return pd.DataFrame(final_pairs)


def clean_and_derive(pairs_df: pd.DataFrame) -> pd.DataFrame:
    pairs_df = pairs_df[(pairs_df["duration"] > 0.03) & (pairs_df["duration"] < 2.0)].dropna()
    pairs_df["log_duration"] = np.log(pairs_df["duration"])
    pairs_df["intensity_centered"] = (pairs_df["intensity"]
                                      - pairs_df.groupby("speaker")["intensity"].transform("mean"))
    pairs_df["f0_avail"] = pairs_df.apply(
        lambda x: x["f0_k"] / x["f0_n"] if x["f0_n"] > 0 else 0, axis=1)
    return pairs_df


# ── 出力 ────────────────────────────────────────────────────────────
def write_outputs(pairs_df: pd.DataFrame, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    pairs_df.to_csv(os.path.join(out_dir, "data_for_r.csv"), index=False)

    print("Generating outputs...")
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({"text.color": "black", "axes.labelcolor": "black",
                         "xtick.color": "black", "ytick.color": "black"})

    table1 = (pairs_df.groupby("condition")[["duration", "log_duration",
                                             "intensity_centered", "f0_avail"]]
              .agg(["mean", "std", "count"]).T)
    table1.to_csv(os.path.join(out_dir, "Table1_Descriptive_Stats.csv"))

    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=pairs_df, x="log_duration", hue="condition", fill=True,
                palette=["gray", "black"], alpha=0.4)
    plt.title("Figure 1: Distribution of Log Duration", fontweight="bold")
    plt.savefig(os.path.join(out_dir, "Figure1_Distribution.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    sns.pointplot(data=pairs_df, x="condition", y="log_duration",
                  order=["first", "repeat"], capsize=.1, color="black", markers="s")
    sns.stripplot(data=pairs_df, x="condition", y="log_duration",
                  order=["first", "repeat"], color="gray", alpha=0.1, jitter=True)
    plt.title("Figure 2: Mean Log Duration", fontweight="bold")
    plt.savefig(os.path.join(out_dir, "Figure2_Comparison.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    sns.boxplot(data=pairs_df, x="condition", y="f0_avail", order=["first", "repeat"],
                showfliers=False, color="white", linecolor="black")
    sns.stripplot(data=pairs_df, x="condition", y="f0_avail", order=["first", "repeat"],
                  color="gray", alpha=0.1, jitter=True)
    plt.title("Figure S1: F0 Availability", fontweight="bold")
    plt.savefig(os.path.join(out_dir, "FigureS1_F0_Availability.png"), dpi=300)
    plt.close()
    print(f"Saved data_for_r.csv, Table1, Figures to {out_dir}")


# ── メイン ──────────────────────────────────────────────────────────
def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)  # specom_restarts/
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=os.path.join(root, "raw"))
    ap.add_argument("--out", default=os.path.join(root, "output"))
    ap.add_argument("--intermediate", default=os.path.join(root, "intermediate"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.intermediate, exist_ok=True)

    freq_dict = load_frequency()
    cand_df = scan_candidates(args.raw)
    if not cand_df.empty:
        cand_df.to_csv(os.path.join(args.intermediate, "cutoff_candidates.csv"), index=False)
    pairs_df = build_pairs(cand_df, args.raw, freq_dict)
    if pairs_df.empty:
        print("No valid pairs found."); return
    pairs_df = clean_and_derive(pairs_df)
    write_outputs(pairs_df, args.out)
    print(f"Done. N={len(pairs_df)} tokens "
          f"({(pairs_df['condition'] == 'first').sum()} first / "
          f"{(pairs_df['condition'] == 'repeat').sum()} repeat).")


if __name__ == "__main__":
    main()
