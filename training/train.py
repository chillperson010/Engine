#!/usr/bin/env python3
"""Train the perspective NNUE on human titled-game positions, with pure WDL
targets (game result only -- no engine evaluation). Exports a quantized .nnue.

    training/.venv/bin/python training/train.py \
        --data data/work/chesscom.txt data/work/lichess_2015-01.txt \
        --epochs 30 --out training/nets/titled-demo.nnue

Targets are the game result from the side-to-move's perspective (win=1, draw=0.5,
loss=0). Loss = MSE(sigmoid(output), target). This is the human-only objective:
the network learns "which side is winning" purely from how titled humans' games
actually ended.
"""
import argparse
import sys
import time

import numpy as np

import nnue_io as N

MAXF = 32  # max pieces on board


def load_data(paths, max_samples=0):
    Fw, Fb, stm, tgt = [], [], [], []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    fen, score = line.split("\t")
                    wf, bf, s = N.parse_fen(fen)
                    white = float(score)
                except Exception:  # noqa: BLE001
                    continue
                if len(wf) > MAXF:
                    continue
                Fw.append(wf)
                Fb.append(bf)
                stm.append(s)
                tgt.append(white if s == 0 else 1.0 - white)  # stm perspective
                if max_samples and len(tgt) >= max_samples:
                    break
    n = len(tgt)
    fw = np.full((n, MAXF), -1, dtype=np.int32)
    fb = np.full((n, MAXF), -1, dtype=np.int32)
    for i in range(n):
        fw[i, : len(Fw[i])] = Fw[i]
        fb[i, : len(Fb[i])] = Fb[i]
    return fw, fb, np.array(stm, np.int8), np.array(tgt, np.float32)


def dense(idx_pad, B):
    """[B,MAXF] padded feature indices -> dense [B,768] one-hot-sum."""
    X = np.zeros((B, N.NUM_FEATURES), dtype=np.float32)
    mask = idx_pad >= 0
    rows = np.broadcast_to(np.arange(B)[:, None], idx_pad.shape)[mask]
    cols = idx_pad[mask]
    X[rows, cols] = 1.0
    return X


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--wd", type=float, default=2e-4, help="decoupled weight decay")
    ap.add_argument("--patience", type=int, default=8, help="early-stop patience (epochs)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"[train] loading {args.data}", file=sys.stderr)
    fw, fb, stm, tgt = load_data(args.data, args.max_samples)
    n = len(tgt)
    if n < 100:
        print(f"[train] only {n} samples -- need more data", file=sys.stderr)
        return 1

    perm = rng.permutation(n)
    fw, fb, stm, tgt = fw[perm], fb[perm], stm[perm], tgt[perm]
    nval = max(1, int(n * args.val_frac))
    tr = slice(nval, n)
    va = slice(0, nval)
    print(f"[train] {n} samples ({n - nval} train / {nval} val); "
          f"draw-baseline MSE={np.mean((tgt[va]-0.5)**2):.4f}", file=sys.stderr)

    Hh = N.H
    W1 = (rng.standard_normal((N.NUM_FEATURES, Hh)) * 0.01).astype(np.float32)
    b1 = np.zeros(Hh, np.float32)
    W2 = (rng.standard_normal(2 * Hh) * 0.01).astype(np.float32)
    b2 = np.float32(0.0)

    # Adam state.
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": np.array(b2)}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    def forward(fwb, fbb, stmb):
        B = len(stmb)
        Xw = dense(fwb, B)
        Xb = dense(fbb, B)
        accw = Xw @ params["W1"] + params["b1"]
        accb = Xb @ params["W1"] + params["b1"]
        is_w = (stmb == 0)[:, None]
        own = np.where(is_w, accw, accb)
        opp = np.where(is_w, accb, accw)
        pre = np.concatenate([own, opp], axis=1)
        hid = np.clip(pre, 0.0, 1.0)
        out = hid @ params["W2"] + params["b2"]
        pred = sigmoid(out)
        return Xw, Xb, accw, accb, pre, hid, out, pred

    def evaluate_loss(sl):
        idxs = range(sl.start, sl.stop)
        losses = []
        for i in range(sl.start, sl.stop, args.batch):
            j = min(i + args.batch, sl.stop)
            *_, pred = forward(fw[i:j], fb[i:j], stm[i:j])
            losses.append(np.sum((pred - tgt[i:j]) ** 2))
        return sum(losses) / (sl.stop - sl.start)
        _ = idxs

    ntr = n - nval
    best_val = float("inf")
    best_params = None
    since_best = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        order = rng.permutation(ntr) + nval
        tot = 0.0
        for s in range(0, ntr, args.batch):
            bidx = order[s : s + args.batch]
            B = len(bidx)
            fwb, fbb, stmb, y = fw[bidx], fb[bidx], stm[bidx], tgt[bidx]
            Xw, Xb, accw, accb, pre, hid, out, pred = forward(fwb, fbb, stmb)
            tot += np.sum((pred - y) ** 2)

            # Backward.
            dout = (2.0 / B) * (pred - y) * pred * (1.0 - pred)         # [B]
            dW2 = hid.T @ dout                                          # [2H]
            db2 = np.sum(dout)
            dhid = np.outer(dout, params["W2"])                        # [B,2H]
            dpre = dhid * ((pre > 0.0) & (pre < 1.0))                  # clip grad
            is_w = (stmb == 0)[:, None]
            d_own = dpre[:, :Hh]
            d_opp = dpre[:, Hh:]
            d_accw = np.where(is_w, d_own, d_opp)
            d_accb = np.where(is_w, d_opp, d_own)
            dW1 = Xw.T @ d_accw + Xb.T @ d_accb                        # [768,H]
            db1 = np.sum(d_accw + d_accb, axis=0)

            grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": np.array(db2)}
            step += 1
            for k in params:
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
                mhat = m[k] / (1 - beta1 ** step)
                vhat = v[k] / (1 - beta2 ** step)
                params[k] -= args.lr * mhat / (np.sqrt(vhat) + eps)
                if args.wd and k in ("W1", "W2"):
                    params[k] -= args.lr * args.wd * params[k]   # decoupled weight decay
            # Keep weights in a quantization-safe range.
            np.clip(params["W1"], -3.0, 3.0, out=params["W1"])
            np.clip(params["W2"], -8.0, 8.0, out=params["W2"])

        val = evaluate_loss(va)
        tag = ""
        if val < best_val - 1e-5:
            best_val = val
            best_params = {k: v.copy() for k, v in params.items()}
            since_best = 0
            tag = " *"
        else:
            since_best += 1
        print(f"[train] epoch {epoch+1:3d}/{args.epochs}  train_mse={tot/ntr:.4f}  "
              f"val_mse={val:.4f}  ({time.time()-t0:.1f}s){tag}", file=sys.stderr)
        if since_best >= args.patience:
            print(f"[train] early stop (no val improvement in {args.patience} epochs)", file=sys.stderr)
            break

    p = best_params or params
    print(f"[train] best val_mse={best_val:.4f}", file=sys.stderr)
    N.write_nnue(args.out, p["W1"], p["b1"], p["W2"], float(p["b2"]))
    print(f"[train] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
