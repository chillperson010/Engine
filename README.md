# TitledNNUE — a UCI chess engine trained only on human games

TitledNNUE is a [UCI](https://en.wikipedia.org/wiki/Universal_Chess_Interface)
chess engine that loads in common chess GUIs (Cute Chess, Arena, BanksiaGUI,
ChessBase, …). It uses a strong classical **alpha-beta search** whose
**evaluation (NNUE) is trained exclusively on human games** — and for online
games, only on games where **both players are titled** (GM/IM/FM/…), so that
engine/cheater play never enters the training signal.

> The search algorithm is unrestricted (that's what makes it strong); only the
> *learned evaluation* is restricted to human data.

## Status

| Milestone | What | State |
|-----------|------|-------|
| **M1** | Working UCI engine + hand-crafted (PeSTO) eval, passes perft | ✅ done |
| **M2** | Data pipeline: download → filter titled-vs-titled → labeled positions | ✅ done |
| **M3** | NNUE training (pure human WDL targets) + C++ inference (exact parity) | ✅ done |
| **M4** | Strength work (SEE, richer eval), SPRT harness, scale-up docs | ✅ done |

**Strength note (honest):** the engine is strongest on its classical search +
hand-crafted evaluation, which improves with SEE-ordered search and positional
terms (mobility, bishop pair, passed pawns, rook files, pawn structure). The
human-only NNUE is fully wired and verified, but a pure-WDL net trained only on
(materially balanced) titled-human games under-learns material and is
off-distribution for search nodes, so it is used as an opt-in positional
*correction* (`EvalFile`), not the default. See `docs/SCALE_UP.md` for the path
to make it surpass the baseline.

## Measured results (node-limited self-play A/B, this build)

| Change | Result vs prior build | Elo |
|--------|----------------------|-----|
| Positional eval terms (mobility, bishop pair, passed pawns, rook files, pawn structure) | +43 −24 =33 | **≈ +67** |
| King safety | +42 −41 =57 | ≈ +2 (neutral in fast self-play; expected to help at real time controls / vs humans) |
| Human NNUE as hybrid correction (regularized) | +17 −24 =19 | ≈ −41 (not a clear win → kept opt-in) |

(Elo via `tools/sprt/match_engines.py` / `match.py`. Estimated absolute strength
~2300–2600 on engine rating scales; not yet measured against a calibrated
opponent.)

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Produces `build/titled-nnue`.

## Use

Point any UCI GUI at the `build/titled-nnue` binary, or drive it directly:

```bash
printf 'uci\nposition startpos\ngo depth 12\nquit\n' | ./build/titled-nnue
```

UCI options:
- `Hash` (MiB) — transposition table size.
- `EvalFile` — path to a `.nnue` network (M3). Without one, the engine uses its
  hand-crafted evaluation.
- `UseNNUE`, `Threads` — accepted; single-threaded search in M1.

## Verify

```bash
./build/titled-nnue perft     # move-generation correctness (6 known positions)
./build/titled-nnue bench 12  # fixed-depth node/nps signature
```

## Search features (M1)

Bitboard move generation (vendored [chess-library](https://github.com/Disservin/chess-library)),
iterative deepening with PVS, transposition table, null-move pruning, late move
reductions, reverse-futility / futility / late-move pruning, quiescence with
delta pruning, MVV-LVA + killer + history move ordering, check extensions,
aspiration windows, mate-distance pruning, and UCI time management.

## Training data policy

See [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md). The eval is trained on
Lichess titled-vs-titled games, OTB human databases (TWIC, …), and (optionally)
chess.com titled players, labeled by **game outcome (WDL)** only — no engine
evaluation is used as a training target.
