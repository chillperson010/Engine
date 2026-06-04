"""Shared NNUE definitions: feature indexing, file format, and the exact integer
evaluation. The C++ engine (src/nnue/nnue.cpp) mirrors this byte-for-byte and
arithmetic-for-arithmetic so a trained network evaluates identically in both.

Network: perspective NNUE  (768 -> H) x2 -> 1
  - 768 features per perspective: (rel_color*6 + piece_type)*64 + rel_square
  - two accumulators (white-perspective, black-perspective), shared weights
  - side-to-move's accumulator is "own", the other is "opp"
  - hidden = clipped_relu(concat(own, opp)) ; output = W2 . hidden + b2

Quantization: feature transformer scale QA, output weight scale QB.
  eval_cp = trunc( out_int * SCALE / (QA*QB) )
"""
import struct

import numpy as np

H = 256
NUM_FEATURES = 768
QA = 255
QB = 64
SCALE = 400
MAGIC = b"TND1"

# FEN piece char -> (color, piece_type)  color: 0 white, 1 black; pt: 0..5
_PIECE = {
    "P": (0, 0), "N": (0, 1), "B": (0, 2), "R": (0, 3), "Q": (0, 4), "K": (0, 5),
    "p": (1, 0), "n": (1, 1), "b": (1, 2), "r": (1, 3), "q": (1, 4), "k": (1, 5),
}


def feature_index(color, pt, sq, persp):
    """Feature index for a piece (color,pt) on square sq, from `persp`'s view."""
    rel_color = 0 if color == persp else 1
    rel_sq = sq if persp == 0 else (sq ^ 56)
    return (rel_color * 6 + pt) * 64 + rel_sq


def parse_fen(fen):
    """Return (white_features, black_features, stm). stm: 0 white, 1 black."""
    parts = fen.split()
    board, stm_str = parts[0], parts[1]
    wf, bf = [], []
    for i, row in enumerate(board.split("/")):
        rank_idx = 7 - i  # row 0 is rank 8
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                color, pt = _PIECE[ch]
                sq = rank_idx * 8 + file
                wf.append(feature_index(color, pt, sq, 0))
                bf.append(feature_index(color, pt, sq, 1))
                file += 1
    stm = 0 if stm_str == "w" else 1
    return wf, bf, stm


def write_nnue(path, W1, b1, W2, b2):
    """Quantize float params and write the .nnue file.
    W1: [NUM_FEATURES, H]  b1: [H]  W2: [2H]  b2: scalar
    """
    ft_w = np.round(W1 * QA).astype(np.int16)        # [F, H]
    ft_b = np.round(b1 * QA).astype(np.int16)        # [H]
    out_w = np.round(W2 * QB).astype(np.int16)       # [2H]
    out_b = int(round(float(b2) * QA * QB))          # int32

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<iiiii", H, NUM_FEATURES, SCALE, QA, QB))
        f.write(struct.pack("<i", out_b))
        f.write(ft_w.tobytes())   # row-major: feature 0's H weights, then feature 1...
        f.write(ft_b.tobytes())
        f.write(out_w.tobytes())


def read_nnue(path):
    with open(path, "rb") as f:
        assert f.read(4) == MAGIC, "bad magic"
        h, nf, scale, qa, qb = struct.unpack("<iiiii", f.read(20))
        (out_b,) = struct.unpack("<i", f.read(4))
        ft_w = np.frombuffer(f.read(nf * h * 2), dtype=np.int16).reshape(nf, h).astype(np.int64)
        ft_b = np.frombuffer(f.read(h * 2), dtype=np.int16).astype(np.int64)
        out_w = np.frombuffer(f.read(2 * h * 2), dtype=np.int16).astype(np.int64)
    return dict(H=h, NF=nf, SCALE=scale, QA=qa, QB=qb, out_b=out_b, ft_w=ft_w, ft_b=ft_b, out_w=out_w)


def trunc_div(n, d):
    """Integer division truncating toward zero (matches C++ '/')."""
    q = abs(n) // d
    return q if n >= 0 else -q


def quantized_eval(net, fen):
    """Exact integer evaluation from the side-to-move's perspective (centipawns).
    Mirrors src/nnue/nnue.cpp."""
    wf, bf, stm = parse_fen(fen)
    h = net["H"]
    accw = net["ft_b"].copy()
    accb = net["ft_b"].copy()
    if wf:
        accw = accw + net["ft_w"][wf].sum(axis=0)
        accb = accb + net["ft_w"][bf].sum(axis=0)
    accw = np.clip(accw, 0, net["QA"])
    accb = np.clip(accb, 0, net["QA"])
    own, opp = (accw, accb) if stm == 0 else (accb, accw)
    out = int(net["out_b"])
    out += int((own * net["out_w"][:h]).sum())
    out += int((opp * net["out_w"][h:]).sum())
    return trunc_div(out * net["SCALE"], net["QA"] * net["QB"])
