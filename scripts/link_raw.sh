#!/bin/bash
# =====================================================================
# link_raw.sh — raw/ をフラットなシンボリックリンク農場として（再）構築
#
# パイプラインは raw/ に .txt / .words / .wav がフラットに並ぶことを想定する
# （meta の README_structure.txt 準拠）。実体は複製せず、正典 Buckeye
#   ~/Projects/research/corpora/buckeye/sNN/sNNSSx.*
# への絶対パス・シンボリックリンクを張る（他プロジェクトと同方式・ライセンス遵守）。
#
# 使い方:  ./link_raw.sh            # 既定の正典から raw/ を再構築
#          ./link_raw.sh /path/to/buckeye
# =====================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"                       # specom_restarts/
RAW="$ROOT/raw"
SRC="${1:-$HOME/Projects/research/corpora/buckeye}"

if [ ! -d "$SRC" ]; then
  echo "ERROR: canonical Buckeye not found at: $SRC" >&2
  exit 1
fi

mkdir -p "$RAW"
n=0
# sNN/ 配下の各収録ファイルをフラットにリンク（必要拡張子のみ）
for ext in txt words wav phones log; do
  while IFS= read -r -d '' f; do
    ln -sf "$f" "$RAW/$(basename "$f")"
    n=$((n + 1))
  done < <(find "$SRC" -mindepth 2 -maxdepth 2 -name "*.$ext" -print0)
done
echo "Linked $n files into $RAW (symlinks → $SRC)"
