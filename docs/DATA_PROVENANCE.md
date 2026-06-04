# Training data provenance & policy

The NNUE evaluation is trained **only on human games**. The goal is to keep
engine play and online cheating out of the training signal.

## Inclusion rules

A game is eligible for training only if it satisfies one of:

1. **Lichess titled-vs-titled (online).** From the monthly Lichess open database
   dumps (`https://database.lichess.org/`), keep a game iff **both** players are
   titled and **neither is a bot**:
   - `[WhiteTitle]` and `[BlackTitle]` are both present, and
   - both titles ∈ {GM, IM, FM, CM, NM, WGM, WIM, WFM, WCM}, and
   - neither title is `BOT`.
   - Honorary Lichess `LM` is **excluded by default** (toggle in the extractor).

2. **Over-the-board human databases.** TWIC / pgnmentor / Caissabase games.
   These are real human OTB play (no online-cheating vector). Title/Elo sanity
   filters are applied where headers allow.

3. **chess.com titled players (optional).** Via the public API: enumerate titled
   usernames (`/pub/titled/{GM,IM,...}`), pull each player's monthly archives,
   and keep only games where **both** usernames are in the titled set.

## Labeling

- Each sampled position is labeled with the **game result (WDL)** from the
  side-to-move's perspective: win / draw / loss.
- **No engine evaluation is used as a training target.** In `nnue-pytorch` terms
  the trainer runs with `lambda = 1.0` (game result only). A flag is left in
  place for a future variant that blends engine scores, but the default and the
  shipped network are strictly human-derived.
- The human move actually played may be recorded as a *move-ordering hint* only;
  it is never fed to the evaluation network.

## Sampling

- Skip the first N opening plies (book-like, low information).
- Skip positions where the side to move is in check (keep "quiet" samples).
- Deduplicate positions; cap positions per game to avoid over-weighting long
  games.

## Reproducibility

The extraction step emits a manifest recording source files, filter counts
(games scanned / kept, positions written), date ranges, and the exact filter
settings used, so any network can be traced back to its data.
