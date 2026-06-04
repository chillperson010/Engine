#!/usr/bin/env python3
"""Play two engine binaries against each other (node-limited) and report the
Elo difference of A relative to B. Used to A/B-test engine changes.

    training/.venv/bin/python tools/sprt/match_engines.py \
        --a build/titled-nnue --b /tmp/baseline-engine --games 200 --nodes 60000
"""
import argparse
import math
import sys

import chess
import chess.engine

OPENINGS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    "rnbqkb1r/pppppppp/5n2/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkbnr/ppp1pppp/8/3p4/2P5/8/PP1PPPPP/RNBQKBNR w KQkq d6 0 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkb1r/pppp1ppp/5n2/4p3/2P5/2N5/PP1PPPPP/R1BQKBNR w KQkq - 0 3",
]


def elo(score, n):
    if score <= 0 or score >= 1:
        return (float("-inf") if score <= 0 else float("inf")), 0.0
    e = -400 * math.log10(1 / score - 1)
    se = math.sqrt(score * (1 - score) / n)
    lo, hi = max(1e-9, score - 1.96 * se), min(1 - 1e-9, score + 1.96 * se)
    return e, (-400 * math.log10(1 / hi - 1) - (-400 * math.log10(1 / lo - 1))) / 2


def play(a, b, fen, a_white, lim, max_plies=240):
    board = chess.Board(fen)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        eng = a if ((board.turn == chess.WHITE) == a_white) else b
        mv = eng.play(board, lim).move
        if mv is None:
            break
        board.push(mv)
    r = board.result(claim_draw=True)
    if r == "1-0":
        return 1.0 if a_white else 0.0
    if r == "0-1":
        return 0.0 if a_white else 1.0
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--nodes", type=int, default=60000)
    ap.add_argument("--movetime", type=int, default=0, help="ms per move (overrides --nodes)")
    ap.add_argument("--a-threads", type=int, default=1)
    ap.add_argument("--b-threads", type=int, default=1)
    args = ap.parse_args()

    a = chess.engine.SimpleEngine.popen_uci(args.a)
    b = chess.engine.SimpleEngine.popen_uci(args.b)
    a.configure({"Hash": 64, "Threads": args.a_threads})
    b.configure({"Hash": 64, "Threads": args.b_threads})
    lim = (chess.engine.Limit(time=args.movetime / 1000.0) if args.movetime
           else chess.engine.Limit(nodes=args.nodes))
    w = l = d = 0
    try:
        for g in range(args.games):
            fen = OPENINGS[(g // 2) % len(OPENINGS)]
            s = play(a, b, fen, g % 2 == 0, lim)
            w += s == 1.0
            l += s == 0.0
            d += s == 0.5
            n = w + l + d
            e, err = elo((w + 0.5 * d) / n, n)
            print(f"[{n:4d}] A +{w} -{l} ={d}  Elo={e:+.0f}+-{err:.0f}", file=sys.stderr)
    finally:
        a.quit()
        b.quit()
    n = w + l + d
    e, err = elo((w + 0.5 * d) / n, n)
    print(f"\nFINAL A vs B: +{w} -{l} ={d}  Elo {e:+.0f} +- {err:.0f}")


if __name__ == "__main__":
    sys.exit(main())
