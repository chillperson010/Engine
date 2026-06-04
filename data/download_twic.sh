#!/usr/bin/env bash
# Download a range of TWIC (The Week in Chess) issues — high-quality OTB human
# games (overwhelmingly titled players, no online-cheating vector) — and extract
# WDL-labeled positions. TWIC ships zipped PGNs named twic<NNNN>g.zip.
#
# Usage:
#   data/download_twic.sh FIRST LAST [OUT_PREFIX]
# Env:
#   MIN_ELO   require both Elo >= this when titles are absent (default 2200)
#   EXTRACT   path to extract binary (default build/extract)
#   EXTRA     extra flags for extract
#
# Example (issues 1500..1510):
#   data/download_twic.sh 1500 1510
set -euo pipefail

FIRST="${1:?usage: download_twic.sh FIRST LAST [OUT_PREFIX]}"
LAST="${2:?usage: download_twic.sh FIRST LAST [OUT_PREFIX]}"
OUT_PREFIX="${3:-data/work/twic_${FIRST}_${LAST}}"
MIN_ELO="${MIN_ELO:-2200}"
EXTRACT="${EXTRACT:-build/extract}"
EXTRA="${EXTRA:-}"

command -v unzip >/dev/null || { echo "need 'unzip' (apt-get install unzip)"; exit 1; }

mkdir -p "$(dirname "$OUT_PREFIX")" data/raw/twic
OUT="${OUT_PREFIX}.txt"
MAN="${OUT_PREFIX}.manifest"
: > "$OUT"

TMP_PGN="$(mktemp)"
trap 'rm -f "$TMP_PGN"' EXIT

for n in $(seq "$FIRST" "$LAST"); do
  url="https://theweekinchess.com/zips/twic${n}g.zip"
  zip="data/raw/twic/twic${n}g.zip"
  echo "[twic] ${url}" >&2
  if curl -fsSL -o "$zip" "$url"; then
    unzip -p "$zip" >> "$TMP_PGN" 2>/dev/null || true
  else
    echo "[twic] issue ${n} not available, skipping" >&2
  fi
done

# OTB: titles aren't always present, so gate on rating instead.
# shellcheck disable=SC2086
"$EXTRACT" --no-require-titles --min-elo "$MIN_ELO" --manifest "$MAN" $EXTRA \
  < "$TMP_PGN" > "$OUT"

echo "[twic] done:" >&2
cat "$MAN" >&2
wc -l "$OUT" >&2
