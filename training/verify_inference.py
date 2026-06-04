#!/usr/bin/env python3
"""Cross-check: the C++ engine's NNUE eval must equal the trainer's exact integer
eval for every FEN. Any mismatch means the quantization/format/feature-indexing
drifted between Python and C++.

    training/.venv/bin/python training/verify_inference.py \
        training/nets/smoke.nnue build/titled-nnue data/work/chesscom.txt
"""
import subprocess
import sys

import nnue_io as N


def main(net_path, engine, data_path, n=200):
    net = N.read_nnue(net_path)

    fens = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            fens.append(line.split("\t")[0])
            if len(fens) >= n:
                break

    # C++ side.
    proc = subprocess.run(
        [engine, "evalfen", net_path],
        input="\n".join(fens) + "\n",
        capture_output=True, text=True, check=True,
    )
    cpp = {}
    for line in proc.stdout.splitlines():
        if "\t" in line:
            fen, val = line.rsplit("\t", 1)
            cpp[fen] = int(val)

    mismatches = 0
    for fen in fens:
        py = N.quantized_eval(net, fen)
        # C++ clamps to +-29000; mirror that for comparison.
        py = max(-29000, min(29000, py))
        c = cpp.get(fen)
        if c != py:
            mismatches += 1
            if mismatches <= 10:
                print(f"  MISMATCH py={py} cpp={c}  {fen}", file=sys.stderr)

    print(f"checked {len(fens)} FENs: {mismatches} mismatches")
    return 1 if mismatches else 0


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: verify_inference.py <net.nnue> <engine> <dataset.txt>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(*sys.argv[1:4]))
