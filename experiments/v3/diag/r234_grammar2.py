"""Round 234: a second, harder controlled grammar for the parse study (gold trees known exactly).
Same lexicon, agreement and selectional restrictions as r230_grammar, plus three constructions that break "bracket the subject, then
branch right":
  NP  -> Det (Adj) N [ "with" Det (Adj) N ]             noun modifier; "with" always attaches to the noun phrase   (p = .2)
  NP_subj -> NP [ "that" VP_rc ]                          subject relative clause, verb agrees with the head noun      (p = .3)
  VP  -> V-frame as r230, then [ "at" Det (Adj) N_place ] verb adverbial; "at" always attaches to the verb phrase   (p = .25)
Each preposition attaches one way only, so the tree is a function of the string. Sentences of 3 to 14 words.
Gold constituents (multi-word, below the sentence): every NP (with its modifier and relative clause), every PP ("with"/"at"/"to"/"in"
phrase), the relative clause, every VP (main and relative). The flat Det (Adj) N has no internal bracket; a VP with an object and an
adverbial has no bracket between them (either binarisation is right), so neither is scored.
load_corpus(seed, n) and gold(s) mirror r230_grammar / r233_goldparse; selected by env R233_GRAMMAR=g2 in the r233 parse scripts."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r230_grammar as G  # noqa: E402

TAG = dict(G.TAG, **{"with": "IN", "at": "IN", "that": "WDT"})
_GOLD = {}


def _np(types, rng, num=None, mod=True):
    """tokens, number, spans (relative to start) of a noun phrase; mod allows a 'with' modifier."""
    toks, num = G._np(types, rng, num)
    spans = []
    if mod and rng.random() < 0.2:
        pp, _ = G._np({"object", "animal", "food"}, rng)
        start = len(toks)
        if start >= 2:
            spans += [(0, start)]                      # the head NP inside NP -> NP PP
        toks = toks + ["with"] + pp
        spans += [(start, len(toks))]                  # PP
        if len(pp) >= 2:
            spans += [(start + 1, len(toks))]          # its NP
        spans += [(0, len(toks))]                      # the modified NP
    elif len(toks) >= 2:
        spans += [(0, len(toks))]
    return toks, num, spans


def _vp(v, num, rng, rc=False):
    """tokens, spans (relative) of a verb phrase headed by verb v with number num."""
    sg, fr, subj, obj, prep = G.VERBS[v]
    toks = [sg if num == "sg" else v]
    spans = []
    if fr == "vt":
        o, _, sp = _np(obj, rng, mod=not rc)
        spans += [(1 + a, 1 + b) for a, b in sp]
        toks += o
    elif fr == "vpp":
        o, _, sp = _np(obj, rng, mod=not rc)
        spans += [(2 + a, 2 + b) for a, b in sp]
        toks += [prep] + o
        spans += [(1, len(toks))]                      # PP
    if not rc and rng.random() < 0.25:
        p, _, sp = _np({"place"}, rng, mod=False)
        start = len(toks)
        toks += ["at"] + p
        spans += [(start + 1 + a, start + 1 + b) for a, b in sp] + [(start, len(toks))]
    if len(toks) >= 2:
        spans += [(0, len(toks))]                      # VP
    return toks, spans


def _verb(fr_w, subj_ok, rng):
    while True:
        fr = rng.choices(["vi", "vt", "vpp"], weights=fr_w)[0]
        cands = [k for k, x in G.VERBS.items() if x[1] == fr and subj_ok(x[2])]
        if cands:
            return G._zipf(cands, rng, 0.4)


def sample_sentence(rng):
    """(tokens, gold span set)."""
    while True:
        v = _verb([0.25, 0.5, 0.25], lambda s: True, rng)
        sg, fr, subj, obj, prep = G.VERBS[v]
        sub, num, ssp = _np(subj, rng)
        spans = list(ssp)
        if "with" not in sub and rng.random() < 0.3:
            # relative clause on the subject: its verb must accept the head noun's type
            head = G.LEX[sub[-1]] if sub[-1] in G.LEX else None
            hty = head[1] if head else None
            rv = _verb([0.3, 0.5, 0.2], lambda s: hty in s, rng)
            if rv != v:
                rtoks, rsp = _vp(rv, num, rng, rc=True)
                start = len(sub)
                sub = sub + ["that"] + rtoks
                spans += [(start + 1 + a, start + 1 + b) for a, b in rsp] + [(start, len(sub)), (0, len(sub))]
        vtoks, vsp = _vp(v, num, rng)
        off = len(sub)
        toks = sub + vtoks
        spans += [(off + a, off + b) for a, b in vsp]
        if 3 <= len(toks) <= 14:
            g = {(a, b) for a, b in spans if 2 <= b - a < len(toks)}
            return toks, g


def load_corpus(seed, n, n_max=4000):
    rng = random.Random(1000 + seed)
    stream = [sample_sentence(rng) for _ in range(max(n, n_max))]
    r2 = random.Random(2000 + seed)
    held = [sample_sentence(r2) for _ in range(1500)]
    for s, g in stream + held:
        prev = _GOLD.setdefault(tuple(s), g)
        assert prev == g, ("ambiguous gold", s)
    return {"train": [(s, False) for s, _ in stream[:n]], "held": [s for s, _ in held]}


def gold(s):
    return _GOLD[tuple(s)]


if __name__ == "__main__":
    rng = random.Random(0)
    from collections import Counter
    L = Counter()
    for i in range(12):
        s, g = sample_sentence(rng)
        print(" ".join(s), sorted(g))
    for _ in range(4000):
        s, g = sample_sentence(rng)
        L[len(s)] += 1
    print(sorted(L.items()))
    C = load_corpus(13, 4000)
    print("ok", len(C["train"]), len(C["held"]))
