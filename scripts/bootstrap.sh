#!/usr/bin/env bash
# Prepare the engine for development/testing: build it and (best-effort) set up
# the Python training venv. Safe to run repeatedly. Used by the SessionStart hook
# and handy locally.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "[bootstrap] configuring + building engine"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
if cmake --build build -j"$(nproc)" >/tmp/engine_build.log 2>&1; then
  echo "[bootstrap] build OK -> build/titled-nnue"
else
  echo "[bootstrap] build FAILED (see /tmp/engine_build.log)"; tail -20 /tmp/engine_build.log
fi

# Optional Python tooling (data pipeline + trainer). Don't fail the session if
# the network is unavailable.
if [ ! -d training/.venv ]; then
  echo "[bootstrap] creating training venv"
  python3 -m venv training/.venv >/dev/null 2>&1 \
    && training/.venv/bin/pip install -q --upgrade pip setuptools wheel >/dev/null 2>&1 \
    && training/.venv/bin/pip install -q numpy chess >/dev/null 2>&1 \
    && echo "[bootstrap] venv ready" \
    || echo "[bootstrap] venv setup skipped (offline?)"
fi

command -v zstd >/dev/null || echo "[bootstrap] note: install 'zstd' to stream Lichess dumps"
echo "[bootstrap] done"
