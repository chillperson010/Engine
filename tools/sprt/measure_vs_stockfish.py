#!/usr/bin/env python3
"""Estimate our engine's Elo by playing it against Stockfish capped at several
UCI_Elo levels and computing a performance rating.

    training/.venv/bin/python tools/sprt/measure_vs_stockfish.py \
        --engine build/titled-nnue --sf /usr/games/stockfish \
        --elos 2200 2400 2600 --games 30 --movetime 100 --threads 1
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
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
]


def play(ours, sf, fen, ours_white, lim, max_plies=240):
    board = chess.Board(fen)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        eng = ours if ((board.turn == chess.WHITE) == ours_white) else sf
        mv = eng.play(board, lim).move
        if mv is None:
            break
        board.push(mv)
    r = board.result(claim_draw=True)
    if r == "1-0":
        return 1.0 if ours_white else 0.0
    if r == "0-1":
        return 0.0 if ours_white else 1.0
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--sf", required=True)
    ap.add_argument("--elos", type=int, nargs="+", required=True)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--movetime", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    lim = chess.engine.Limit(time=args.movetime / 1000.0)
    perfs = []
    weights = []
    for sf_elo in args.elos:
        ours = chess.engine.SimpleEngine.popen_uci(args.engine)
        sf = chess.engine.SimpleEngine.popen_uci(args.sf)
        ours.configure({"Hash": 64, "Threads": args.threads})
        sf.configure({"Hash": 64, "Threads": 1, "UCI_LimitStrength": True, "UCI_Elo": sf_elo})
        w = l = d = 0
        try:
            for g in range(args.games):
                fen = OPENINGS[(g // 2) % len(OPENINGS)]
                s = play(ours, sf, fen, g % 2 == 0, lim)
                w += s == 1.0
                l += s == 0.0
                d += s == 0.5
        finally:
            ours.quit()
            sf.quit()
        n = w + l + d
        score = (w + 0.5 * d) / n
        sc = min(max(score, 1e-6), 1 - 1e-6)
        perf = sf_elo + 400 * math.log10(sc / (1 - sc))
        perfs.append(perf)
        weights.append(n)
        print(f"vs SF@{sf_elo}: +{w} -{l} ={d}  score={score:.3f}  -> perf {perf:.0f}",
              file=sys.stderr)

    est = sum(p * wt for p, wt in zip(perfs, weights)) / sum(weights)
    print(f"\nESTIMATED Elo (perf-rating avg, movetime={args.movetime}ms, "
          f"threads={args.threads}): {est:.0f}")


if __name__ == "__main__":
    sys.exit(main())
