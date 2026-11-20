"""Round 231: word-tree representations that keep ALL neighbours (no function-word-only features). Changes only the attributes fed to WordTree's Cobweb tree.
feats (WordTree factory `make`):
  all        : the original WordTree (left / right neighbour identities).
  abs:B      : two passes. Pass 1 = the original 'all' tree; each word's class at level B bits. Pass 2 = same L / R facets, but a neighbour is
               named by its pass-1 class at level B (sentence markers stay), so the word is described by the classes of ALL its neighbours.
  pos        : facets L, R (immediate neighbours, identities) and W (the wider context, positions +-2 and +-3, identities), all neighbours kept.
  allsuf     : the original L, R facets plus a suffix facet S (last 1-3 characters), added not substituted.
Unsupervised: counts over the raw token stream only."""
from typing import Dict

from trellis2.trellis3_r214 import WordTree as _WT


class WordTree231(_WT):
    def __init__(self, sentences, seed: int = 0, feats: str = "pos"):
        from trellis2.cu_cobweb import CUCobwebTree
        from trellis2.trellis3 import BOS, EOS
        sentences = [list(s) for s in sentences]
        cname = None
        if feats.startswith("abs:"):
            B = float(feats.split(":")[1])
            cname = _WT(sentences, seed, "all").names(b=B)
        facets: Dict[str, Dict[str, Dict[str, float]]] = {"L": {}, "R": {}}
        if feats == "pos":
            facets["W"] = {}
        if feats == "allsuf":
            facets["S"] = {}
        order = []

        def nm(x):
            return cname.get(x, x) if cname is not None else x

        for s in sentences:
            t = [BOS, BOS, BOS] + s + [EOS, EOS, EOS]
            for i in range(3, len(t) - 3):
                w = t[i]
                if w not in facets["L"]:
                    order.append(w)
                    for f in facets:
                        facets[f][w] = {}
                    if feats == "allsuf":
                        for k in (1, 2, 3):
                            if len(w) > k:
                                facets["S"][w]["%d:%s" % (k, w[-k:])] = 1.0
                for f, j in (("L", i - 1), ("R", i + 1)):
                    d = facets[f][w]
                    d[nm(t[j])] = d.get(nm(t[j]), 0.0) + 1.0
                if feats == "pos":
                    for j in (i - 2, i - 3, i + 2, i + 3):
                        d = facets["W"][w]
                        d[t[j]] = d.get(t[j], 0.0) + 1.0
        self.feats = feats
        norm = lambda d: {k: v / sum(d.values()) for k, v in d.items()} if d else {"-": 1.0}
        self.tree = CUCobwebTree(list(facets), seed=seed)
        self.tree.exact_alone = True
        self.leaf = {w: self.tree.ifit({f: norm(facets[f][w]) for f in facets}) for w in order}
        self.words = order


def make(sentences, seed=0, feats="all"):
    if feats == "all" or feats.startswith("func:"):
        return _WT(sentences, seed, feats)
    return WordTree231(sentences, seed, feats)
