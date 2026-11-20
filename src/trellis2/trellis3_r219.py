"""Round 219: word-tree feature sets that carry agreement and frame cues (a representation change of the existing WordTree; no new generation machinery).

WordTree219(sentences, seed, feats) is WordTree with a richer description of each word TYPE. feats = "fn:K[+d2][+lr][+nb]":
  fn:K  the K most frequent word types (plus sentence markers) are the feature vocabulary (the function-word contexts; as "func:K" of r214)
  +d2   two more facets, the distribution of the word two to the left and two to the right (agreement and frames are decided across one word,
        e.g. "was [adj]", "the [adj] cat was")
  +lr   one more facet, the joint (left, right) pair of feature words: "a _ was" marks a singular count noun, "to _ the" a transitive verb
  +nb   the facets use K only for the neighbour word kept; a type with none of its neighbours in the vocabulary keeps the dummy "-"
Learning is unchanged (the same Fisher Cobweb tree over bags of probabilities); levels b are read as before.
"""
from __future__ import annotations

from typing import Dict

from .trellis3_r214 import WordTree


class WordTree219(WordTree):
    def __init__(self, sentences, seed: int = 0, feats: str = "fn:50"):
        from .cu_cobweb import CUCobwebTree
        from .trellis3 import BOS, EOS
        parts = feats.split("+")
        K = int(parts[0].split(":")[1])
        d2, lr = "d2" in parts[1:], "lr" in parts[1:]
        freq: Dict[str, int] = {}
        for s in sentences:
            for w in s:
                freq[w] = freq.get(w, 0) + 1
        keep = {w for w, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:K]} | {BOS, EOS}
        facets = {"L": {}, "R": {}}
        if d2:
            facets.update({"L2": {}, "R2": {}})
        if lr:
            facets["LR"] = {}
        order = []
        seen = set()

        def add(f, w, k, v=1.0):
            d = facets[f].setdefault(w, {})
            if k in keep or f == "LR":
                d[k] = d.get(k, 0.0) + v

        for s in sentences:
            t = [BOS, BOS] + list(s) + [EOS, EOS]
            for i in range(2, len(t) - 2):
                w = t[i]
                if w not in seen:
                    seen.add(w)
                    order.append(w)
                    for f in facets:
                        facets[f][w] = {}
                add("L", w, t[i - 1])
                add("R", w, t[i + 1])
                if d2:
                    add("L2", w, t[i - 2])
                    add("R2", w, t[i + 2])
                if lr and t[i - 1] in keep and t[i + 1] in keep:
                    add("LR", w, t[i - 1] + "|" + t[i + 1])
        self.feats = feats
        norm = lambda d: {k: v / sum(d.values()) for k, v in d.items()} if d else {"-": 1.0}
        self.tree = CUCobwebTree(list(facets), seed=seed)
        self.tree.exact_alone = True
        self.leaf = {w: self.tree.ifit({f: norm(facets[f][w]) for f in facets}) for w in order}
        self.words = order
