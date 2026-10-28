"""Chess state compression with a lossless chunk dictionary (CHREST-style).

Stage A coded a recognised chunk by its content concept, but a content description
holds the parts' learned names, not their pieces, so that code cannot be decoded
back to the board. Here a chunk is coded only as a pointer to an identical stored
chunk (the same pieces on the same squares), so every code decodes to the position.

  dictionary  the composites memory chose for the ``n_learn`` training positions,
              each keyed by its exact (square, piece) set
  coding      a position's parse is read top-down: a unit whose key is in the
              dictionary is one item; otherwise its two parts are coded in turn; a
              single piece is one piece-square item
  bits        items are entropy-coded with their usage frequencies on the training
              positions (Laplace, plus one stop item per position)
  baseline    the same code with no chunks: piece-square items only
  dictionary  storing it costs each used chunk written as a piece-square list; it is
              pruned to chunks chosen in at least m training positions, m = 1 .. 20

Also reported: the share of the units recognised at ``theta`` (as chess_recall.py)
that have an identical stored chunk. Writes an npz that stage B reads (``trellis``
= item counts, lossless) with the labels of an earlier stage-A run.

Usage: chess_compress_dict.py <cfg json> <seed> <tag> <stage-A npz> [n_rep n_learn n_train n_test theta]
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2 import chess as C  # noqa: E402
from trellis2.trellis_graph import GraphTrellis  # noqa: E402

cfg, seed, tag, a_npz = json.loads(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
n_rep, n_learn, n_train, n_test = [int(x) for x in sys.argv[5:9]] if len(sys.argv) > 8 else [4000, 2000, 2500, 1000]
theta = float(sys.argv[9]) if len(sys.argv) > 9 else 12.0
max_size = cfg.pop("max_size", 4)

fens = [l.strip() for l in open(os.path.join(ROOT, "data", "chess", "settings", "positions_1800_ply30.fen")) if l.strip()]
random.Random(seed).shuffle(fens)
rep_fens = fens[:n_rep]
lab_fens = fens[n_rep:n_rep + n_train + n_test]
A = np.load(a_npz)
assert int(A["n_rep"]) == n_rep and int(A["n_train"]) == n_train and len(A["values"]) == len(lab_fens)
KINDS = sorted(C.TOKENS)


def key(e, exp):
    sqs = exp.data[1]
    return frozenset((sqs[i], exp.tokens[i]) for i in e.span)


def ps(e, exp):
    (i,) = tuple(e.span)
    f, r = exp.data[1][i]
    return KINDS.index(exp.tokens[i]) * 64 + r * 8 + f


# ---------------------------------------------------------------- learn, collect the dictionary
L = GraphTrellis(max_size=max_size, seed=seed, **cfg)
store = Counter()
t0 = time.time()
for f in rep_fens[:n_learn]:
    r = L.learn(C.parse_fen(f))
    for k in r["chosen"]:
        store[key(r["comps"][k], r["exp"])] += 1
learn_s = time.time() - t0


def climb(e):
    if "climb" not in e.memo:
        e.memo["climb"] = L._climb(L.cnt, L._content(e)) + L._climb(L.ctx, e.ctx)
    return e.memo["climb"]


def units(e):
    if e.token is not None or climb(e) >= theta:
        return [e]
    return units(e.left) + units(e.right)


def parse_once(fen):
    """(the parse's top units and its experience, recognised units, of them with a stored twin)"""
    r = L.parse(C.parse_fen(fen))
    exp, rec, twin = r["exp"], 0, 0
    for top in r["top"]:
        for u in units(top):
            if u.token is None:
                rec += 1
                twin += key(u, exp) in store
    return r["top"], exp, rec, twin


def items(parsed, D):
    """The items of the lossless code with dictionary D (a set of chunk keys)."""
    tops, exp = parsed[0], parsed[1]
    out = []

    def code(e):
        if e.token is not None:
            out.append(("p", ps(e, exp)))
            return
        k = key(e, exp)
        if k in D:
            out.append(("c", k))
            return
        code(e.left)
        code(e.right)
    for top in tops:
        code(top)
    return out


t1 = time.time()
train_parsed = [parse_once(f) for f in rep_fens[:n_learn]]
lab_parsed = [parse_once(f) for f in lab_fens]
code_s = time.time() - t1


def coder(counts):
    alphabet = set(counts) | {("p", i) for i in range(768)}
    tot = sum(counts.values()) + n_learn + len(alphabet) + 1      # + stop items, Laplace
    cost = lambda it: -math.log2((counts.get(it, 0) + 1) / tot)
    stop = -math.log2((n_learn + 1) / tot)
    return cost, stop


use_raw = Counter(("p", KINDS.index(t) * 64 + r * 8 + fl) for f in rep_fens[:n_learn] for (fl, r), t in C.parse_fen(f).items())
cost_r, stop_r = coder(use_raw)
rbits = np.array([sum(cost_r(("p", KINDS.index(t) * 64 + r * 8 + fl)) for (fl, r), t in C.parse_fen(f).items()) + stop_r
                  for f in lab_fens], dtype=np.float32)
n_lab = len(lab_fens)
summary = {"tag": tag, "seed": seed, "theta": theta, "max size": max_size, "learn s": round(learn_s), "code s": round(code_s),
           "dictionary chunks": len(store),
           "piece-square bits mean (held out)": round(float(rbits.mean()), 1),
           "uniform piece-square bits mean": round(float(A["boards"][n_rep:].sum(1).mean() * math.log2(768)), 1),
           "recognised units per position": round(sum(p[2] for p in lab_parsed) / n_lab, 2),
           "recognised units with an identical stored chunk": round(sum(p[3] for p in lab_parsed) / max(sum(p[2] for p in lab_parsed), 1), 3),
           "by minimum use": []}
best = None
for m in (1, 2, 3, 5, 10, 20):
    # the dictionary: chunks memory chose in at least m training positions
    D = {k for k, c in store.items() if c >= m}
    train_items = [items(p, D) for p in train_parsed]
    lab_items = [items(p, D) for p in lab_parsed]
    use = Counter(it for its in train_items for it in its)
    cost_d, stop_d = coder(use)
    used = sorted({it for it in use if it[0] == "c"}, key=lambda it: sorted(it[1]))
    # storing the dictionary: each chunk it uses written as a piece-square list
    dict_bits = sum(sum(cost_r(("p", KINDS.index(t) * 64 + r * 8 + fl)) for (fl, r), t in it[1]) + stop_r for it in used)
    dbits = np.array([sum(cost_d(it) for it in its) + stop_d for its in lab_items], dtype=np.float32)
    row = {"min use": m, "chunks": len(used), "bits per position": round(float(dbits.mean()), 1),
           "dictionary bits": round(dict_bits), "dictionary bits per held-out position": round(dict_bits / n_lab, 1),
           "total per position (1,000 held out)": round(float(dbits.mean()) + dict_bits / n_lab, 1),
           "break-even positions": round(dict_bits / max(float(rbits.mean() - dbits.mean()), 1e-9)) if dbits.mean() < rbits.mean() else None,
           "items per position": round(float(np.mean([len(i) for i in lab_items])), 2),
           "chunks per position": round(float(np.mean([sum(it[0] == "c" for it in i) for i in lab_items])), 2)}
    summary["by minimum use"].append(row)
    print(json.dumps(row), flush=True)
    if m == 1:
        vocab = {it: j for j, it in enumerate(used)}
        feats = np.zeros((n_lab, len(vocab) + 768), dtype=np.float32)
        for j, its in enumerate(lab_items):
            for it in its:
                if it[0] == "c" and it in vocab:
                    feats[j, vocab[it]] += 1.0
                elif it[0] == "c":                   # in the dictionary but never used in training: its pieces
                    for (fl, r), t in it[1]:
                        feats[j, len(vocab) + KINDS.index(t) * 64 + r * 8 + fl] = 1.0
                else:
                    feats[j, len(vocab) + it[1]] = 1.0
        best = (feats, dbits, np.array([len(i) for i in lab_items], dtype=np.float32))

out_dir = os.path.join(ROOT, "experiments", "v3", "results", "chess_compress")
np.savez_compressed(os.path.join(out_dir, f"{tag}_s{seed}.npz"), boards=A["boards"], n_rep=n_rep, n_train=n_train,
                    values=A["values"], trellis=best[0], trellis_bits=best[1], trellis_units=best[2], raw_coded_bits=rbits)
json.dump(summary, open(os.path.join(out_dir, f"{tag}_s{seed}.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(summary), flush=True)
