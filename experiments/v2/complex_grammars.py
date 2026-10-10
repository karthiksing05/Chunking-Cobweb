"""Complex synthetic grammars: what real syntax has and the paper's six lack.

Three grammars of graded difficulty, each strictly binary + lexical (the form
``trellis2.evaluation.CFG`` recognizes and ``util.cfg.generate_with_merges``
expands into gold merges):

* ``agree``    number agreement (determiner-noun, subject-verb, and a
               coordinated NP is plural), adjectives, PPs on nouns and verbs,
               coordination of NPs and of VPs; 6-15 words.
* ``relative`` ``agree`` plus relative clauses on nouns: subject relatives
               (right-branching, the relative's verb agrees with the head
               noun: "the dog that chases the cats runs") and object
               relatives (centre embedding: "the dog that the cats chase
               runs"), which nest; 8-20 words.
* ``deep``     ``relative`` plus reduced object relatives ("the dog the cat
               chases runs"), sentential complements ("the girl thinks that
               the dogs run"), sentence coordination ("..., but ..."), a larger
               lexicon; 10-30 words.

Number is carried by split nonterminals (``_s`` singular, ``_p`` plural), so
agreement is a context-free fact the recognizer checks exactly. "the" is both
a singular and a plural determiner, and "that" is both a relativizer and a
complementizer (``deep``), as in English.

Production weights are integers: ``util.cfg.generate_with_merges`` draws a
production uniformly from the list, so a production of weight w is listed w
times (each word of a lexical class once). Recursion is kept subcritical and
sentences outside the length range are rejected.

Corpora (``make``): 2,000 distinct training sentences and 500 distinct held-out
sentences not among them, in ``data/cfg_complex_<name>/{train,heldout}/``,
one JSON file ``{"sentence", "merges"}`` per sentence as in the paper's
corpora (``trellis2.data.load_corpus`` reads them; files are numbered, so the
order is the generation order).

Usage:
    python experiments/v2/complex_grammars.py --make            # writes data/cfg_complex_*
    python experiments/v2/complex_grammars.py --check           # recognizer and agreement checks
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trellis2.data import load_corpus  # noqa: E402
from trellis2.evaluation import CFG  # noqa: E402
from util.cfg import generate_with_merges  # noqa: E402

# --------------------------------------------------------------------------- #
# Lexicons

LEX_SMALL = {
    "Det_s": ["the", "a", "every", "this"],
    "Det_p": ["the", "some", "these", "many"],
    "N": [("dog", "dogs"), ("cat", "cats"), ("girl", "girls"), ("boy", "boys"),
          ("bird", "birds"), ("teacher", "teachers")],
    "Vt": [("sees", "see"), ("chases", "chase"), ("likes", "like"), ("helps", "help"),
           ("follows", "follow")],
    "Vi": [("runs", "run"), ("sleeps", "sleep"), ("laughs", "laugh"), ("sings", "sing")],
    "Adj": ["big", "small", "old", "happy"],
    "P": ["near", "with", "behind", "under"],
}

LEX_LARGE = {
    "Det_s": ["the", "a", "every", "this", "each"],
    "Det_p": ["the", "some", "these", "many", "few"],
    "N": [("dog", "dogs"), ("cat", "cats"), ("girl", "girls"), ("boy", "boys"),
          ("bird", "birds"), ("teacher", "teachers"), ("farmer", "farmers"), ("king", "kings"),
          ("horse", "horses"), ("child", "children")],
    "Vt": [("sees", "see"), ("chases", "chase"), ("likes", "like"), ("helps", "help"),
           ("follows", "follow"), ("finds", "find"), ("knows", "know"), ("watches", "watch")],
    "Vi": [("runs", "run"), ("sleeps", "sleep"), ("laughs", "laugh"), ("sings", "sing"),
           ("waits", "wait"), ("smiles", "smile")],
    "Vc": [("thinks", "think"), ("says", "say"), ("hopes", "hope")],
    "Adj": ["big", "small", "old", "happy", "tall", "clever"],
    "P": ["near", "with", "behind", "under", "beside", "above"],
}


def _lexicon(lex: dict) -> Dict[str, List[List[str]]]:
    g = {"Det_s": [[w] for w in lex["Det_s"]], "Det_p": [[w] for w in lex["Det_p"]],
         "Vt_s": [[s] for s, _ in lex["Vt"]], "Vt_p": [[p] for _, p in lex["Vt"]],
         "Adj": [[w] for w in lex["Adj"]], "P": [[w] for w in lex["P"]], "And": [["and"]]}
    if "Vc" in lex:
        g["Vc_s"] = [[s] for s, _ in lex["Vc"]]
        g["Vc_p"] = [[p] for _, p in lex["Vc"]]
    return g


def _weighted(*items: Tuple[List[str], int]) -> List[List[str]]:
    out = []
    for rhs, w in items:
        out += [list(rhs)] * w
    return out


def build(name: str) -> Dict[str, List[List[str]]]:
    """The grammar ``name`` in util.cfg's form: lhs -> list of productions
    (duplicates are weights); a one-element production that is not a
    nonterminal is a word."""
    lex = LEX_LARGE if name == "deep" else LEX_SMALL
    g = _lexicon(lex)
    relative = name in ("relative", "deep")
    deep = name == "deep"
    nouns = {"s": [s for s, _ in lex["N"]], "p": [p for _, p in lex["N"]]}
    vis = {"s": [s for s, _ in lex["Vi"]], "p": [p for _, p in lex["Vi"]]}
    n_noun, n_vi = len(lex["N"]), len(lex["Vi"])
    # Sentences: subject and verb agree in number.
    g["S"] = _weighted((["NP_s", "VP_s"], 6), (["NP_p", "VP_p"], 5))
    if deep:
        g["S"] += _weighted((["S", "CoS"], 1))
        g["CoS"] = [["But", "S"]]
        g["But"] = [["but"]]
    for x in "sp":
        # Noun phrases: a determiner of the noun's number and a noun group.
        g[f"NP_{x}"] = _weighted(([f"Det_{x}", f"Nb_{x}"], 12))
        # Noun groups: a noun (lexical), adjectives, PPs, relative clauses.
        nb = [[w] for w in nouns[x]]
        nb += _weighted((["Adj", f"Nb_{x}"], round(0.25 * n_noun)),
                        ([f"Nb_{x}", "PP"], round(0.15 * n_noun)))
        if relative:
            nb += _weighted(([f"Nb_{x}", f"RC_{x}"], round(0.22 * n_noun)))
        if deep:
            nb += _weighted(([f"Nb_{x}", "Gap"], round(0.08 * n_noun)))
        g[f"Nb_{x}"] = nb
        # Verb phrases: an intransitive verb (lexical), a transitive verb and
        # its object, a PP, coordinated VPs (same number), complements.
        vp = [[w] for w in vis[x]]
        vp += _weighted(([f"Vt_{x}", "NP_s"], round(0.45 * n_vi)), ([f"Vt_{x}", "NP_p"], round(0.35 * n_vi)),
                        ([f"VP_{x}", "PP"], round(0.12 * n_vi)), ([f"VP_{x}", f"CoVP_{x}"], round(0.10 * n_vi)))
        if deep:
            vp += _weighted(([f"Vc_{x}", "Comp"], round(0.25 * n_vi)))
        g[f"VP_{x}"] = vp
        g[f"CoVP_{x}"] = [["And", f"VP_{x}"]]
        if relative:
            # Subject relative (the relative's verb agrees with the head
            # noun), or object relative with its gap (centre embedding).
            g[f"RC_{x}"] = _weighted((["Rel", f"VP_{x}"], 3), (["Rel", "Gap"], 2))
    # A coordinated NP is plural, whatever its parts.
    g["NP_p"] += _weighted((["NP_s", "CoNP"], 3), (["NP_p", "CoNP"], 1))
    g["CoNP"] = _weighted((["And", "NP_s"], 1), (["And", "NP_p"], 1))
    g["PP"] = _weighted((["P", "NP_s"], 3), (["P", "NP_p"], 2))
    if relative:
        g["Rel"] = [["that"], ["who"]]
        # The object gap: the relative's own subject and its transitive verb, which agree.
        g["Gap"] = _weighted((["NP_s", "Vt_s"], 1), (["NP_p", "Vt_p"], 1))
    if deep:
        g["Comp"] = [["That", "S"]]
        g["That"] = [["that"]]
    return g


def number_blind(name: str) -> Dict[str, List[List[str]]]:
    """The same grammar with number erased: every ``X_s`` / ``X_p`` becomes
    ``X`` (words keep their forms). A sentence it accepts but ``build``
    rejects fails only on agreement; one it rejects fails on structure."""
    out: Dict[str, List[List[str]]] = {}
    strip = lambda x: x[:-2] if x.endswith(("_s", "_p")) else x   # noqa: E731
    g = build(name)
    for lhs, prods in g.items():
        rows = out.setdefault(strip(lhs), [])
        for p in prods:
            q = [strip(x) if x in g else x for x in p]
            if q not in rows:
                rows.append(q)
    return out


GRAMMARS = ("agree", "relative", "deep")
LENGTHS = {"agree": (6, 15), "relative": (8, 20), "deep": (10, 30)}
# Corpus seeds, one per grammar.
SEEDS = {"agree": 101, "relative": 202, "deep": 303}


def data_dir(name: str) -> str:
    return os.path.join(ROOT, "data", f"cfg_complex_{name}")


def load(name: str):
    """(training examples in generation order, held-out examples)."""
    d = data_dir(name)
    return load_corpus(os.path.join(d, "train")), load_corpus(os.path.join(d, "heldout"))


# --------------------------------------------------------------------------- #

def _check_gold(cfg: CFG, tokens, tree) -> bool:
    """The gold tree is a parse of the grammar: every span's category set,
    built bottom-up through the tree's own brackets, holds S at the root."""
    labels = {}
    for (i, j) in tree.bottom_up():
        if j - i == 1:
            labels[(i, j)] = cfg._close(set(cfg.lex.get(tokens[i], ())))
        else:
            k = tree.split[(i, j)]
            cell = set()
            for b in labels[(i, k)]:
                for c in labels[(k, j)]:
                    cell |= cfg.binary.get((b, c), set())
            labels[(i, j)] = cfg._close(cell)
    return cfg.start in labels[(0, tree.n)]


def make(name: str, n_train: int = 2000, n_heldout: int = 500, max_attempts: int = 2_000_000):
    g = build(name)
    lo, hi = LENGTHS[name]
    random.seed(SEEDS[name])
    sys.setrecursionlimit(10000)
    seen, kept = set(), []
    attempts = too_long = too_short = 0
    while len(kept) < n_train + n_heldout and attempts < max_attempts:
        attempts += 1
        try:
            text, merges = generate_with_merges("S", g, flatten_at_parent=())
        except RecursionError:
            too_long += 1
            continue
        text = text.strip()
        n = len(text.split())
        if n > hi:
            too_long += 1
            continue
        if n < lo:
            too_short += 1
            continue
        if text in seen:
            continue
        seen.add(text)
        kept.append((text, merges))
    if len(kept) < n_train + n_heldout:
        raise RuntimeError(f"{name}: only {len(kept)} distinct sentences")
    d = data_dir(name)
    for part, items, offset in (("train", kept[:n_train], 0), ("heldout", kept[n_train:], n_train)):
        os.makedirs(os.path.join(d, part), exist_ok=True)
        for old in glob.glob(os.path.join(d, part, "*.json")):
            os.remove(old)
        for i, (text, merges) in enumerate(items):
            h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
            with open(os.path.join(d, part, f"{part}_{i:04d}_{h}.json"), "w", encoding="utf-8") as f:
                json.dump({"sentence": text, "merges": merges}, f)
    print(f"{name}: {len(kept)} sentences from {attempts} draws ({too_long} too long, "
          f"{too_short} too short) -> {d}", flush=True)


def check(name: str) -> dict:
    """Recognizer checks: every sentence is recognized and its gold tree is a
    parse; agreement violations (one verb's or noun's number flipped) are
    rejected unless the flip happens to leave a grammatical sentence."""
    g = build(name)
    cfg = CFG(g)
    train, held = load(name)
    out = {"grammar": name, "train": len(train), "heldout": len(held)}
    allx = train + held
    out["recognized"] = sum(cfg.recognizes(e.tokens) for e in allx) / len(allx)
    out["gold tree is a parse"] = sum(_check_gold(cfg, e.tokens, e.tree) for e in allx) / len(allx)
    lens = [len(e.tokens) for e in allx]
    out["length min/mean/max"] = (min(lens), sum(lens) / len(lens), max(lens))
    out["vocabulary"] = len({w for e in allx for w in e.tokens})
    out["nonterminals"] = len([k for k in g if any(len(p) == 2 or p[0] in g for p in g[k])])
    out["held-out sentences also in training"] = len({e.sentence for e in held} & {e.sentence for e in train})
    # Agreement violations: flip the number of one verb (or noun) form.
    lex = LEX_LARGE if name == "deep" else LEX_SMALL
    flip = {}
    for cls in ("N", "Vt", "Vi", "Vc"):
        for s, p in lex.get(cls, []):
            flip[s], flip[p] = p, s
    rng = random.Random(0)
    rejected = total = 0
    for e in held[:300]:
        pos = [i for i, w in enumerate(e.tokens) if w in flip]
        if not pos:
            continue
        i = rng.choice(pos)
        bad = list(e.tokens)
        bad[i] = flip[bad[i]]
        total += 1
        rejected += not cfg.recognizes(bad)
    out["one number flipped: rejected"] = rejected / max(total, 1)
    # Constructions present (from the gold trees' categories).
    counts = Counter()
    for e in train:
        s = " " + e.sentence + " "
        counts["coordination 'and'"] += " and " in s
        if name != "agree":
            counts["relativizer"] += (" that " in s) or (" who " in s)
    out["sentences with"] = {k: v / len(train) for k, v in counts.items()}
    return out


def bracketed(tokens, tree) -> str:
    return tree.to_string(tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--grammars", default=",".join(GRAMMARS))
    args = ap.parse_args()
    for name in args.grammars.split(","):
        if args.make:
            make(name)
        if args.check:
            r = check(name)
            print(json.dumps(r, indent=1))
            train, _ = load(name)
            for e in train[:6]:
                print("   ", bracketed(e.tokens, e.tree))


if __name__ == "__main__":
    main()
