#!/usr/bin/env python3
"""Train the human move-prediction (policy) network on (position, human-move)
pairs from titled games. Predicts the from-square and to-square a titled human
would play. Used by the engine for human-style move ordering and root selection.

    build/extract --policy --no-require-titles < data/work/chesscom_big.pgn > data/work/policy.txt
    training/.venv/bin/python training/train_policy.py --data data/work/policy.txt \
        --epochs 30 --out training/nets/human-policy.pol
"""
import argparse
import sys
import time

import numpy as np

import nnue_io as N
import policy_io as P

MAXF = 32


def load(paths, max_samples=0):
    F, rf, rt = [], [], []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line or "\t" not in line:
                    continue
                fen, uci = line.split("\t")
                if len(uci) < 4:
                    continue
                try:
                    feats, a, b = P.parse_sample(fen, uci)
                except Exception:  # noqa: BLE001
                    continue
                if not feats or len(feats) > MAXF:
                    continue
                F.append(feats)
                rf.append(a)
                rt.append(b)
                if max_samples and len(rf) >= max_samples:
                    break
    n = len(rf)
    fp = np.full((n, MAXF), -1, dtype=np.int32)
    for i in range(n):
        fp[i, : len(F[i])] = F[i]
    return fp, np.array(rf, np.int64), np.array(rt, np.int64)


def dense(idx_pad, B):
    X = np.zeros((B, N.NUM_FEATURES), np.float32)
    mask = idx_pad >= 0
    rows = np.broadcast_to(np.arange(B)[:, None], idx_pad.shape)[mask]
    X[rows, idx_pad[mask]] = 1.0
    return X


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    fp, rf, rt = load(args.data, args.max_samples)
    n = len(rf)
    if n < 200:
        print(f"only {n} samples", file=sys.stderr)
        return 1
    perm = rng.permutation(n)
    fp, rf, rt = fp[perm], rf[perm], rt[perm]
    nval = max(1, int(n * args.val_frac))
    print(f"[policy] {n} samples ({n-nval} train / {nval} val)", file=sys.stderr)

    Hp = P.HP
    W1 = (rng.standard_normal((N.NUM_FEATURES, Hp)) * 0.01).astype(np.float32)
    b1 = np.zeros(Hp, np.float32)
    Wf = (rng.standard_normal((Hp, 64)) * 0.01).astype(np.float32)
    bf = np.zeros(64, np.float32)
    Wt = (rng.standard_normal((Hp, 64)) * 0.01).astype(np.float32)
    bt = np.zeros(64, np.float32)
    params = {"W1": W1, "b1": b1, "Wf": Wf, "bf": bf, "Wt": Wt, "bt": bt}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    b1c, b2c, eps = 0.9, 0.999, 1e-8
    step = 0

    def forward(Fb):
        X = dense(Fb, len(Fb))
        pre = X @ params["W1"] + params["b1"]
        hid = np.maximum(pre, 0.0)
        flog = hid @ params["Wf"] + params["bf"]
        tlog = hid @ params["Wt"] + params["bt"]
        return X, hid, flog, tlog

    def topk_acc(sl):
        i0, i1 = sl.start, sl.stop
        fc = tc = tot = 0
        for i in range(i0, i1, args.batch):
            j = min(i + args.batch, i1)
            _, _, fl, tl = forward(fp[i:j])
            fc += (fl.argmax(1) == rf[i:j]).sum()
            tc += (tl.argmax(1) == rt[i:j]).sum()
            tot += j - i
        return fc / tot, tc / tot

    ntr = n - nval
    for epoch in range(args.epochs):
        t0 = time.time()
        order = rng.permutation(ntr) + nval
        for s in range(0, ntr, args.batch):
            bidx = order[s : s + args.batch]
            B = len(bidx)
            Fb, af, at = fp[bidx], rf[bidx], rt[bidx]
            X, hid, flog, tlog = forward(Fb)
            pf = softmax(flog)
            pt = softmax(tlog)
            dF = pf.copy(); dF[np.arange(B), af] -= 1.0; dF /= B
            dT = pt.copy(); dT[np.arange(B), at] -= 1.0; dT /= B
            dWf = hid.T @ dF; dbf = dF.sum(0)
            dWt = hid.T @ dT; dbt = dT.sum(0)
            dhid = dF @ params["Wf"].T + dT @ params["Wt"].T
            dpre = dhid * (hid > 0)
            dW1 = X.T @ dpre; db1 = dpre.sum(0)
            grads = {"W1": dW1, "b1": db1, "Wf": dWf, "bf": dbf, "Wt": dWt, "bt": dbt}
            step += 1
            for k in params:
                m[k] = b1c * m[k] + (1 - b1c) * grads[k]
                v[k] = b2c * v[k] + (1 - b2c) * grads[k] ** 2
                params[k] -= args.lr * (m[k] / (1 - b1c ** step)) / (np.sqrt(v[k] / (1 - b2c ** step)) + eps)
        fa, ta = topk_acc(slice(0, nval))
        print(f"[policy] epoch {epoch+1:3d}/{args.epochs}  from_acc={fa:.3f} to_acc={ta:.3f} "
              f"({time.time()-t0:.1f}s)", file=sys.stderr)

    P.write_policy(args.out, params["W1"], params["b1"], params["Wf"], params["bf"],
                   params["Wt"], params["bt"])
    print(f"[policy] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
