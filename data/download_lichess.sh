#!/usr/bin/env bash
# Stream a monthly Lichess standard dump, filter to titled-vs-titled games, and
# write WDL-labeled training positions. The full dumps are tens of GB, so by
# default we cap how much we download (MAX_MB) and how many qualifying games we
# keep (MAX_GAMES) — enough for the M3 demonstrator. Raise both to scale up.
#
# Usage:
#   data/download_lichess.sh YYYY-MM [OUT_PREFIX]
# Env:
#   MAX_MB     compressed bytes to fetch from the dump   (default 300)
#   MAX_GAMES  stop after this many kept games           (default 4000; 0 = no cap)
#   EXTRACT    path to the extract binary                (default build/extract)
#   EXTRA      extra flags passed through to extract
#
# Examples:
#   data/download_lichess.sh 2024-01
#   MAX_MB=4000 MAX_GAMES=200000 data/download_lichess.sh 2024-01 data/work/lichess
set -euo pipefail

MONTH="${1:?usage: download_lichess.sh YYYY-MM [OUT_PREFIX]}"
OUT_PREFIX="${2:-data/work/lichess_${MONTH}}"
MAX_MB="${MAX_MB:-300}"
MAX_GAMES="${MAX_GAMES:-4000}"
EXTRACT="${EXTRACT:-build/extract}"
EXTRA="${EXTRA:-}"

URL="https://database.lichess.org/standard/lichess_db_standard_rated_${MONTH}.pgn.zst"
mkdir -p "$(dirname "$OUT_PREFIX")"
OUT="${OUT_PREFIX}.txt"
MAN="${OUT_PREFIX}.manifest"

BYTES=$(( MAX_MB * 1024 * 1024 ))
echo "[lichess] streaming up to ${MAX_MB} MiB of ${URL}" >&2
echo "[lichess] keeping up to ${MAX_GAMES} titled-vs-titled games -> ${OUT}" >&2

# curl a byte range (partial .zst) -> zstd decodes available frames (errors at
# the truncation, which we ignore) -> extract filters + labels.
# shellcheck disable=SC2086
curl -fsSL --range "0-${BYTES}" "$URL" \
  | { zstd -dc 2>/dev/null || true; } \
  | "$EXTRACT" --max-games "$MAX_GAMES" --manifest "$MAN" $EXTRA \
  > "$OUT"

echo "[lichess] done:" >&2
cat "$MAN" >&2
wc -l "$OUT" >&2
