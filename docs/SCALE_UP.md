# Scaling up to a strong human-trained network

The repository ships a **working engine**, the **complete data + training
pipeline**, and a **small demonstrator network**. This document explains how to
grow the demonstrator into a genuinely strong network on your own hardware, and
is honest about what the demonstrator does and does not yet achieve.

## Where the demonstrator stands

- The classical search + hand-crafted PeSTO evaluation (HCE) is already strong
  and is the engine's default.
- The NNUE is trained **only on human titled games** with **pure game-result
  (WDL) targets** — exactly the data policy you asked for. The full training and
  C++ inference paths are verified (the C++ eval matches the trainer bit-for-bit).
- **Honest limitation:** titled-human games are almost always materially
  balanced, so a pure-WDL network sees very few lopsided-material positions and
  therefore *under-learns material values*. On its own it is weaker than PeSTO.
  The engine compensates with a **hybrid evaluation** — material/PST baseline
  (axioms, trained on no games) **plus** the human NNUE as a positional
  correction (`src/eval.cpp`). This keeps material strength while letting the
  human-trained component supply positional judgement.

## Levers, in order of impact

1. **Much more data.** The demonstrator uses ~0.4M positions. Strong NNUEs use
   100M–1B+. Stream many months of Lichess titled-vs-titled, many TWIC issues,
   and a large chess.com titled crawl:
   ```bash
   MAX_MB=8000 MAX_GAMES=0 data/download_lichess.sh 2024-01
   data/download_twic.sh 1400 1600
   training/.venv/bin/python data/fetch_chesscom_titled.py --max-players 2000 --months 24 --out data/work/cc.pgn
   ```
2. **Include decisive / imbalanced positions.** To teach material from human
   games, bias sampling toward decisive games and later phases (lower
   `--sample-every`, raise `--max-per-game`, prefer non-`1/2-1/2` results). This
   is the single most important fix for the material-blindness above.
3. **A real GPU trainer.** The bundled NumPy trainer (`training/train.py`) is for
   CPU demonstration. For scale, port the identical architecture/quantization
   (`training/nnue_io.py`) to PyTorch, or use **bullet**
   (https://github.com/jw1912/bullet) — keep `lambda = 1.0` (pure WDL) to honour
   the human-only policy, or set a small engine-score blend only if you choose to
   relax the policy.
4. **A stronger feature set.** Move from the 768-feature perspective net to
   **HalfKP** or **HalfKAv2_hm** (king-bucketed features). This is the biggest
   architectural step toward top-tier strength.
5. **Incremental accumulator.** `src/nnue/nnue.cpp` currently does a full refresh
   per evaluation (correct, ~0.8M nps). Add an incrementally-updated accumulator
   in make/unmake for a large nps gain (and thus more search depth → strength).
6. **More epochs + LR schedule + larger hidden layer** (`H` in `nnue_io.py`).

## Validate every scale-up

```bash
# 1. data integrity
training/.venv/bin/python data/validate_dataset.py data/work/your.txt
# 2. train
training/.venv/bin/python training/train.py --data data/work/*.txt --epochs 200 --out training/nets/big.nnue
# 3. C++ inference parity (must be 0 mismatches)
training/.venv/bin/python training/verify_inference.py training/nets/big.nnue build/titled-nnue data/work/your.txt
# 4. strength vs the HCE baseline (SPRT)
training/.venv/bin/python tools/sprt/match.py --engine build/titled-nnue --net training/nets/big.nnue --games 1000 --nodes 100000
```

When a network's hybrid configuration SPRT-beats HCE, make it the default by
loading it via the `EvalFile` UCI option (or bake the path into your GUI config).
Until then, the engine plays at its strongest on the HCE default, with the human
network available via `EvalFile` for experimentation.
