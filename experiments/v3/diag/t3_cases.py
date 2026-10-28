"""Case studies for the status page (S1 prompted generation, S2 parse
consistency): worked examples that the measurement scripts do not log.

  S1  gap filling (gap_mode="window", as t3_prompt.py) on fixed prompts, with the
      five most probable fillers and whether each is a word or a chunk, and the
      drawn completion (three draws)
  S2  substitution: a held-out sentence's parse, and the parse after one word is
      replaced by its nearest word by learned name and by a far word (as
      t3_parse_consistency.py)

Usage: t3_cases.py <cfg json> <seed> <n> <tag>
"""
import json
import random
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "experiments/v3/diag")
from trellis2.stories import heldout_path, load_corpus  # noqa: E402
from trellis2.trellis3 import Elem, Trellis3  # noqa: E402

cfg, seed, n, tag = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
corpus = load_corpus(vocab_size=250)
random.Random(seed).shuffle(corpus)
held = json.load(open(heldout_path(13), encoding="utf-8"))
held_set = {tuple(s) for s in held}
L = Trellis3(seed=seed, **cfg)
seen = {}
for s in [s for s in corpus if tuple(s) not in held_set][:n]:
    L.learn(s)
    for w in s:
        seen[w] = seen.get(w, 0) + 1
L._quiet = True


def render(toks, spans):
    """Bracket string from a set of spans (the whole sentence included)."""
    spans = sorted({tuple(s) for s in spans if s[1] - s[0] >= 2} | {(0, len(toks))}, key=lambda s: (s[0], -s[1]))
    opens, closes = {}, {}
    for i, j in spans:
        opens[i] = opens.get(i, 0) + 1
        closes[j] = closes.get(j, 0) + 1
    out = []
    for k, w in enumerate(toks):
        out.append("[" * opens.get(k, 0) + w + "]" * closes.get(k + 1, 0))
    return " ".join(out)


def parse(toks):
    return render(toks, L.parse(toks)["tree"].brackets())


def filler_text(f):
    if f[0] == "w":
        return f[1]
    return " ".join(L._type_key(L.records[f[1]]))


rng = random.Random(seed + 5)
prompts = [(["lily"], []), (["the"], []), (["the", "bird"], []), (["they", "played"], []), (["i", "am"], []),
           (["lily"], ["and", "laughed"]), (["sam"], ["help", "you"]), (["the"], ["the", "tree"]),
           (["i"], ["and", "ben"]), (["it"], ["for", "him"]), (["they"], ["day", "long"])]
for left, right in prompts:
    trace = []
    outs = []
    for _ in range(3):
        g = L.generate_tree(rng, choose="leaf", fill_name="both+peer", gap=(left, right), trace=trace if not outs else None)
        outs.append(" ".join(g) if g else None)
    top = next((t["gap fillers"] for t in trace if "gap fillers" in t), [])
    tot = sum(v for _, v in top) or 1.0
    print(json.dumps({"case": "S1", "tag": tag, "seed": seed, "n": n,
                      "prompt": " ".join(left + ["*"] + right),
                      "top fillers (share of top 10)": [[filler_text(f), "word" if f[0] == "w" else "chunk", round(v / tot, 3)] for f, v in top[:5]],
                      "draws": outs}), flush=True)

# S2 substitution
names_cache = {}


def name(w):
    if w not in names_cache:
        L._rel_cache, L._wc_cache, L._nb_cache = {}, {}, {}
        v = L._ctx_name(Elem((0, 1), token=w)) or {}
        if isinstance(v, list):
            real = [u for u in v if u and u != "-"]
            v = {u: real.count(u) / len(real) for u in set(real)} if real else {}
        t = sum(v.values()) or 1.0
        names_cache[w] = {u: p / t for u, p in v.items()}
    return names_cache[w]


overlap = lambda a, b: sum(min(p, b.get(u, 0.0)) for u, p in a.items())
vocab = [w for w, m in seen.items() if m >= 3]
srng = random.Random(seed + 99)
done = 0
for s in held:
    pos = [i for i, w in enumerate(s) if w in vocab and len(s) >= 5]
    if not pos:
        continue
    i = srng.choice(pos)
    others = sorted((overlap(name(s[i]), name(x)), x) for x in vocab if x != s[i])
    if not others or others[-1][0] <= 0:
        continue
    near, far = others[-1][1], srng.choice([x for _, x in others[:max(1, len(others) // 4)]])
    rows = {"original": parse(s)}
    for label, w in (("near", near), ("far", far)):
        t = list(s)
        t[i] = w
        rows[f"{label} ({s[i]} -> {w})"] = parse(t)
    print(json.dumps({"case": "S2", "tag": tag, "seed": seed, "n": n, **rows}), flush=True)
    done += 1
    if done >= 8:
        break
