#!/usr/bin/env python3
"""Measure how human-like our engine's play is, using the statistical signals
engine-detection/anti-cheat systems rely on:

  * T1 match %  -- how often our move equals a strong engine's top choice
  * ACPL        -- average centipawn loss vs a strong engine's evaluation

Humans (even strong GMs) show lower match% and higher ACPL than engines. This
script compares configurations (e.g. HumanStyle off vs on) so you can see whether
the human-policy mode shifts the statistics toward the human range.

NOTE: this is a move-statistics proxy. Real anti-cheat also uses move timing,
account history and behavioural signals, which this does not model. It measures
human-likeness for research, not evasion.

    training/.venv/bin/python tools/analysis/humanlikeness.py \
        --engine build/titled-nnue --sf /usr/games/stockfish \
        --policy training/nets/human-policy.pol \
        --styles 0 60 --games 10 --our-movetime 100 --opp-elo 2200 --analyzer-depth 12
"""
import argparse
import sys

import chess
import chess.engine

OPENINGS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
]


def cp(score, pov):
    # Cap mate scores so they don't dominate; per-move CPL is clamped below too,
    # the way Lichess/chess.com bound centipawn loss.
    return score.pov(pov).score(mate_score=2000)


CPL_CAP = 1000  # ignore "loss" beyond this on any single move (decided positions)


def analyze_game_moves(board_moves, analyzer, depth):
    """board_moves: list of (board_before, move_played_by_us). Returns (matches, n, cpl_sum)."""
    matches = 0
    cpl_sum = 0
    n = 0
    for board, mv in board_moves:
        info = analyzer.analyse(board, chess.engine.Limit(depth=depth))
        best = info["pv"][0]
        pov = board.turn
        best_cp = cp(info["score"], pov)
        board.push(mv)
        info2 = analyzer.analyse(board, chess.engine.Limit(depth=depth))
        played_cp = cp(info2["score"], pov)  # same pov as the mover
        board.pop()
        cpl_sum += min(CPL_CAP, max(0, best_cp - played_cp))
        matches += (mv == best)
        n += 1
    return matches, n, cpl_sum


def play_collect(ours, opp, fen, ours_white, our_lim, max_plies=160):
    """Play one game; return list of (board_before_copy, our_move) for our moves."""
    board = chess.Board(fen)
    ours_moves = []
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        our_turn = (board.turn == chess.WHITE) == ours_white
        if our_turn:
            mv = ours.play(board, our_lim).move
            if mv is None:
                break
            ours_moves.append((board.copy(), mv))
            board.push(mv)
        else:
            mv = opp.play(board, chess.engine.Limit(time=0.05)).move
            if mv is None:
                break
            board.push(mv)
    return ours_moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--sf", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--styles", type=int, nargs="+", default=[0, 60])
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--our-movetime", type=int, default=100)
    ap.add_argument("--opp-elo", type=int, default=2200)
    ap.add_argument("--analyzer-depth", type=int, default=12)
    args = ap.parse_args()

    our_lim = chess.engine.Limit(time=args.our_movetime / 1000.0)
    print(f"{'HumanStyle':>10} | {'moves':>6} | {'T1 match %':>10} | {'ACPL':>6}")
    print("-" * 44)

    for style in args.styles:
        ours = chess.engine.SimpleEngine.popen_uci(args.engine)
        opp = chess.engine.SimpleEngine.popen_uci(args.sf)
        analyzer = chess.engine.SimpleEngine.popen_uci(args.sf)
        ours.configure({"Hash": 64, "Threads": 1, "HumanFile": args.policy, "HumanStyle": style})
        opp.configure({"Hash": 64, "Threads": 1, "UCI_LimitStrength": True, "UCI_Elo": args.opp_elo})
        analyzer.configure({"Hash": 128, "Threads": 1})

        tot_m = tot_n = tot_cpl = 0
        try:
            for g in range(args.games):
                fen = OPENINGS[(g // 2) % len(OPENINGS)]
                mv_list = play_collect(ours, opp, fen, g % 2 == 0, our_lim)
                m, n, c = analyze_game_moves(mv_list, analyzer, args.analyzer_depth)
                tot_m += m
                tot_n += n
                tot_cpl += c
        finally:
            ours.quit()
            opp.quit()
            analyzer.quit()

        if tot_n:
            print(f"{style:>10} | {tot_n:>6} | {100*tot_m/tot_n:>9.1f}% | {tot_cpl/tot_n:>6.1f}",
                  file=sys.stderr)
            print(f"{style:>10} | {tot_n:>6} | {100*tot_m/tot_n:>9.1f}% | {tot_cpl/tot_n:>6.1f}")


if __name__ == "__main__":
    sys.exit(main())
