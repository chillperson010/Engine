#!/usr/bin/env python3
"""Play the NNUE build against the hand-crafted-eval build of the same binary and
report the result with an Elo estimate and an SPRT verdict.

Both sides are the same engine; the only difference is whether an EvalFile (the
trained human-only network) is loaded. Games are node-limited so the comparison
isolates evaluation quality at equal search effort (the NNUE eval is slower per
node; see docs/SCALE_UP.md for the incremental-accumulator speedup).

    training/.venv/bin/python tools/sprt/match.py \
        --engine build/titled-nnue --net training/nets/titled-demo.nnue \
        --games 200 --nodes 80000
"""
import argparse
import math
import sys

import chess
import chess.engine

# A spread of opening positions (played once with each color) for game variety.
OPENINGS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    "rnbqkb1r/pppppppp/5n2/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkbnr/ppp1pppp/8/3p4/2P5/8/PP1PPPPP/RNBQKBNR w KQkq d6 0 2",
    "rnbqkbnr/pppppp1p/6p1/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkb1r/pppppppp/5n2/8/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 2 2",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
]


def elo(score, n):
    """Elo difference and 95% error bar from a score rate."""
    if score <= 0 or score >= 1:
        return float("inf") if score >= 1 else float("-inf"), 0.0
    e = -400 * math.log10(1 / score - 1)
    # crude 95% CI using stddev of the score rate
    var = score * (1 - score) / n
    se = math.sqrt(var)
    lo = max(1e-9, score - 1.96 * se)
    hi = min(1 - 1e-9, score + 1.96 * se)
    e_lo = -400 * math.log10(1 / lo - 1)
    e_hi = -400 * math.log10(1 / hi - 1)
    return e, (e_hi - e_lo) / 2


def sprt_llr(w, l, d, elo0=0.0, elo1=10.0):
    """Simple score-based log-likelihood ratio for H1(elo1) vs H0(elo0).
    For alpha=beta=0.05 the classic accept/reject bounds are +-2.94."""
    if w == 0 or l == 0:
        return 0.0

    def ll(e):
        p = 1 / (1 + 10 ** (-e / 400))
        return w * math.log(p) + l * math.log(1 - p)

    return ll(elo1) - ll(elo0)


def play_game(eng_a, eng_b, start_fen, a_is_white, nodes, max_plies=240):
    board = chess.Board(start_fen)
    limit = chess.engine.Limit(nodes=nodes)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        white_to_move = board.turn == chess.WHITE
        eng = eng_a if (white_to_move == a_is_white) else eng_b
        result = eng.play(board, limit)
        if result.move is None:
            break
        board.push(result.move)
    res = board.result(claim_draw=True)
    if res == "1-0":
        return 1.0 if a_is_white else 0.0
    if res == "0-1":
        return 0.0 if a_is_white else 1.0
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--nodes", type=int, default=80000)
    ap.add_argument("--hash", type=int, default=32)
    args = ap.parse_args()

    a = chess.engine.SimpleEngine.popen_uci(args.engine)  # NNUE
    b = chess.engine.SimpleEngine.popen_uci(args.engine)  # HCE
    a.configure({"Hash": args.hash, "EvalFile": args.net})
    b.configure({"Hash": args.hash})  # no EvalFile -> hand-crafted eval

    w = l = d = 0  # from NNUE (engine A) point of view
    try:
        for g in range(args.games):
            fen = OPENINGS[(g // 2) % len(OPENINGS)]
            a_white = (g % 2 == 0)
            s = play_game(a, b, fen, a_white, args.nodes)
            if s == 1.0:
                w += 1
            elif s == 0.0:
                l += 1
            else:
                d += 1
            n = w + l + d
            score = (w + 0.5 * d) / n
            e, err = elo(score, n)
            llr = sprt_llr(w, l, d)
            print(f"[{n:4d}] NNUE  W{w} L{l} D{d}  score={score:.3f}  "
                  f"Elo={e:+.0f}+-{err:.0f}  LLR={llr:+.2f}", file=sys.stderr)
    finally:
        a.quit()
        b.quit()

    n = w + l + d
    score = (w + 0.5 * d) / n
    e, err = elo(score, n)
    print(f"\nFINAL  NNUE vs HCE: +{w} -{l} ={d}  score={score:.3f}  Elo {e:+.0f} +- {err:.0f}")
    if e - err > 0:
        print("RESULT: NNUE is stronger than the hand-crafted eval (95% CI above 0).")
    elif e + err < 0:
        print("RESULT: NNUE is weaker than the hand-crafted eval.")
    else:
        print("RESULT: inconclusive at this game count.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
