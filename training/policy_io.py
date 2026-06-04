"""Human move-prediction (policy) network: "learn like a human, play like one."

A factored policy over the side-to-move's perspective:
  input 768 features -> hidden Hp (ReLU) -> two heads: P(from square), P(to square)
  move score = from_logit[rel_from] + to_logit[rel_to]

The C++ engine (src/policy.cpp) mirrors this with float math. Exactness is not
required (policy only guides move ordering and a human-style root blend, never a
TT value), so weights are stored as float32 — no quantization.
"""
import struct

import numpy as np

import nnue_io as N  # reuse parse_fen / feature indexing

HP = 128
MAGIC = b"TPL1"


def square_index(s):
    """'e2' -> 0..63 (a1=0)."""
    return (ord(s[1]) - ord("1")) * 8 + (ord(s[0]) - ord("a"))


def parse_sample(fen, uci):
    """Return (perspective_features, rel_from, rel_to) or None if unusable."""
    wf, bf, stm = N.parse_fen(fen)
    feats = wf if stm == 0 else bf
    frm = square_index(uci[0:2])
    to = square_index(uci[2:4])
    rel_from = frm if stm == 0 else (frm ^ 56)
    rel_to = to if stm == 0 else (to ^ 56)
    return feats, rel_from, rel_to


def write_policy(path, W1, b1, Wf, bf, Wt, bt):
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<ii", W1.shape[1], N.NUM_FEATURES))  # Hp, NF
        for arr in (W1, b1, Wf, bf, Wt, bt):
            f.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())


def read_policy(path):
    with open(path, "rb") as f:
        assert f.read(4) == MAGIC, "bad magic"
        hp, nf = struct.unpack("<ii", f.read(8))

        def rd(n):
            return np.frombuffer(f.read(n * 4), dtype=np.float32)

        W1 = rd(nf * hp).reshape(nf, hp)
        b1 = rd(hp)
        Wf = rd(hp * 64).reshape(hp, 64)
        bf = rd(64)
        Wt = rd(hp * 64).reshape(hp, 64)
        bt = rd(64)
    return dict(HP=hp, NF=nf, W1=W1, b1=b1, Wf=Wf, bf=bf, Wt=Wt, bt=bt)
