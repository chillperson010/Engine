#!/usr/bin/env python3
"""Validate an extracted dataset: every FEN must parse and every label must be a
legal WDL value. Run with the venv Python:

    training/.venv/bin/python data/validate_dataset.py data/work/chesscom.txt
"""
import sys

import chess


def main(path):
    n = 0
    bad = 0
    label_counts = {0.0: 0, 0.5: 0, 1.0: 0}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            n += 1
            try:
                fen, score = line.split("\t")
                board = chess.Board(fen)  # raises on malformed FEN
                if not board.is_valid():
                    raise ValueError("invalid position")
                s = float(score)
                if s not in label_counts:
                    raise ValueError(f"bad label {score}")
                label_counts[s] += 1
            except Exception as e:  # noqa: BLE001
                bad += 1
                if bad <= 5:
                    print(f"  bad line {n}: {e}: {line[:60]}", file=sys.stderr)

    print(f"{path}: {n} samples, {bad} invalid")
    print(f"  labels  white-loss={label_counts[0.0]}  draw={label_counts[0.5]}  white-win={label_counts[1.0]}")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: validate_dataset.py <dataset.txt>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
