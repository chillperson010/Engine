# Human-likeness of play (move-quality analysis)

This measures how human-like the engine's *moves* are — purely from move quality,
not timing or behavioural signals. Two metrics, the same ones engine-detection
relies on:

- **ACPL** — average centipawn loss vs a strong reference engine (Stockfish 16,
  fixed depth). Per-move loss is clamped (like Lichess/chess.com) so blunders and
  mates don't dominate. Lower = more accurate.
- **T1 engine-match %** — how often the move equals the reference engine's top
  choice.

Run with `tools/analysis/humanlikeness.py`.

## Results (8 games per setting, our engine 100 ms/move, analyzer depth 10)

| HumanStyle | Moves | Engine-match % | ACPL |
|-----------:|------:|---------------:|-----:|
| 0 (pure strength) | 398 | 45.7% | 33.4 |
| 40 | 515 | 36.9% | 54.5 |
| 80 | 418 | 43.1% | 58.5 |

### Reference ranges

| Profile | ACPL | Match % |
|---|---|---|
| Full-strength engine (cheater signature) | ~5–20 | ~60–85% |
| Super-GM (classical) | ~15–25 | ~50–60% |
| Strong club/expert (~1900–2200) | ~30–60 | ~40–50% |
| Casual (~1300–1700) | ~70–130 | ~30–40% |

## Reading the result

- By move quality, the engine does **not** present an engine signature. Even at
  `HumanStyle=0` it looks like a strong club player (ACPL ~33, match ~46%), and
  `HumanStyle` pushes it further into human territory.
- The main reason is that it is **roughly human-strength** (~2500 @ 100 ms with an
  imperfect hand-crafted eval); it naturally makes human-magnitude errors. A
  full-strength engine would show ACPL ~5 and be obvious.
- `HumanStyle` increases human-likeness by trading away accuracy (higher ACPL),
  i.e. by playing *weaker*. "Human-like at high strength" is therefore a tension:
  staying near top strength keeps ACPL low, which reads more engine-like.

## Caveats / scope

This is a small-sample, two-metric proxy. Production anti-cheat aggregates over
many games and uses signals not modelled here (move timing, where accuracy peaks,
consistency vs rating history). Matching these two statistics does not imply
undetectability.

Intended use: studying human-like play, and deploying the engine **openly as a
registered bot** or sparring partner. Covert use to play one's moves in online
games is cheating and against site terms of service.
