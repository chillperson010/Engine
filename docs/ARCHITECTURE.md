# Architecture

TitledNNUE is a classical alpha-beta engine with a neural (NNUE) evaluation.
Strength comes from deep search; the data restriction (human-only) applies only
to the learned evaluation.

## Components

```
src/
  main.cpp        entry: UCI loop, plus `perft` and `bench` subcommands
  uci.cpp         UCI protocol: handshake, position, go/stop, options, threading
  search.{h,cpp}  iterative deepening, PVS, pruning, quiescence, ordering   <-- strength core
  tt.{h,cpp}      Zobrist transposition table (depth-preferred replacement)
  eval.{h,cpp}    evaluation dispatch + tapered PeSTO hand-crafted eval (HCE)
  nnue/nnue.{h,cpp}  NNUE inference (stub in M1; HalfKP SIMD forward pass in M3)
  timeman.h       time-budget computation from UCI clock info
  types.h         shared score constants
third_party/chess-library/  vendored bitboard movegen + Zobrist + PGN/UCI helpers
```

## Search (src/search.cpp)

- Iterative deepening with aspiration windows.
- Principal Variation Search (PVS) negamax.
- Transposition table cutoffs + best-move ordering.
- Pruning: null-move, reverse-futility (static null move), futility, late-move
  pruning, late move reductions (LMR).
- Quiescence search over captures with MVV-LVA ordering and delta pruning.
- Move ordering: TT move → captures (MVV-LVA) → promotions → killers → history.
- Check extensions, mate-distance pruning, repetition / 50-move / insufficient
  material draw detection.

Move legality is guaranteed: the engine only ever returns a move produced by the
library's `legalmoves` generator.

## Evaluation (src/eval.cpp, src/nnue)

`eng::evaluate()` returns a score from the side-to-move's perspective. It uses
the NNUE network when one is loaded (`nnue::is_loaded()`), otherwise the
hand-crafted tapered PeSTO evaluation. This indirection means M3 can drop in the
real network with no changes to search.

### NNUE plan (M3)

- HalfKP feature set, perspective feature transformer with an incrementally
  updated accumulator (only a few features change per make/unmake).
- Quantized int8/int16 forward pass with AVX2 SIMD.
- `.nnue` networks trained with `nnue-pytorch` using **pure game-result (WDL)
  targets** (`lambda=1.0`) so the eval is derived solely from human outcomes.

## Why this is strong despite a human-only eval

Alpha-beta with iterative deepening reaches deep into the tree; minimax backup
propagates the consequences of an imperfect leaf eval through tactical lines, so
the evaluation only needs good *relative* judgement of quiet positions. Deep
search with a decent human-trained eval is far stronger than shallow search with
a perfect eval.
