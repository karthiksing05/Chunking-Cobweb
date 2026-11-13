"""Reference coders on characters given the reader's parse state (round 168).

A character's IDS is a tree written in prefix order. A left-to-right reader knows, at every
position, which operator slot the next token fills (or that the tree is complete, so the end
mark is next). The plain n-gram does not track that; the chart does, through its grammar.
This script asks how much of the chart's lead over the causal context route that state is
worth: Witten-Bell n-grams (as chars_coders.ngram) whose every context also carries the slot,
backing off (slot, l1..lk) -> ... -> (slot) -> (). Reference only, never read by TRELLIS.

  python experiments/v3/chars_slot_ngram.py <seed> <n,n,...> [out.json]
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from trellis2.characters import default_ids_path, load_characters, token_slots  # noqa: E402

seed, sizes = int(sys.argv[1]), [int(x) for x in sys.argv[2].split(",")]
out = sys.argv[3] if len(sys.argv) > 3 else None
N_TEST = int(os.environ.get("N_TEST", 300))
chars, _ = load_characters(default_ids_path(), seed=seed)
test = [c.tokens for c in chars[-N_TEST:] if len(c.tokens) > 1]
inventory = {t for c in chars for t in c.tokens}
ESC = math.log2(len(inventory))


def slot_of(s):
    # token_slots gives None for the root and for operators; keep the operator's own slot apart
    out_, st = [], []
    from trellis2.characters import ARITY
    st.append("root")
    for t in s:
        out_.append(st.pop() if st else "overflow")
        if t in ARITY:
            st.extend(reversed([(t, k) for k in range(ARITY[t])]))
    out_.append("end" if not st else "open")
    return out_


def ctx_list(seq, sl, i, order, use_slot):
    """Back-off contexts, low to high: (), (l1), (l2 l1) ...; with the slot: (), (slot), (slot l1), ..."""
    if not use_slot:
        return [tuple(seq[i - k:i]) for k in range(order)]
    s_ = sl[i - order + 1]
    return [()] + [(s_,) + tuple(seq[i - k:i]) for k in range(order)]


def coder(train, order, use_slot):
    counts = [defaultdict(Counter) for _ in range(order + 1)]   # one level more with the slot
    for s in train:
        seq = ["<s>"] * (order - 1) + list(s) + ["</s>"]
        sl = slot_of(s)
        for i in range(order - 1, len(seq)):
            ctxs = ctx_list(seq, sl, i, order, use_slot)
            for lev, c in enumerate(ctxs):
                counts[lev][c][seq[i]] += 1
    V = len(inventory) + 1
    seen = {t for s in train for t in s}

    def bits(s):
        seq = ["<s>"] * (order - 1) + list(s) + ["</s>"]
        sl = slot_of(s)
        b = 0.0
        for i in range(order - 1, len(seq)):
            ctxs = ctx_list(seq, sl, i, order, use_slot)
            pr = 1.0 / V
            for lev, c in enumerate(ctxs):
                d = counts[lev].get(c)
                if d:
                    tot, types = sum(d.values()), len(d)
                    pr = (d.get(seq[i], 0) + types * pr) / (tot + types)
            b += -math.log2(pr)
        return b + sum(ESC for t in s if t not in seen)
    return bits


rows = []
for n in sizes:
    train = [c.tokens for c in chars[:n]]
    row = {"seed": seed, "train": n}
    for order in (2, 3, 4):
        for use in (False, True):
            f = coder(train, order, use)
            row[f"{order}-gram{' + slot' if use else ''}"] = sum(f(s) for s in test) / len(test)
    rows.append(row)
    print(json.dumps(row), flush=True)
if out:
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
