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
| **M2** | Data pipeline: download → filter titled-vs-titled → labeled positions | ⏳ |
| **M3** | NNUE training (pure human WDL targets) + C++ SIMD inference | ⏳ |
| **M4** | Strength testing (SPRT), packaging, scale-up docs | ⏳ |

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
