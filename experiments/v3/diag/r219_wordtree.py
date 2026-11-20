"""Round 219/222: word-tree class dump without any TRELLIS learning (the tree only needs the stream). Cheap (CPU), env trellis.
  r219_wordtree.py <ref wordtree json (gives the sentences)> <out json> <feats: all | func:K | fn:K[+d2][+lr][+suf][+nf]> <seed>
The out json has the r214 wordtree format (sentences, names for b in 0..7 and leaf) plus, r222, size cuts "m<k>": a word's class is the lowest
ancestor of its leaf holding at least k member types (a read-out rule on the same tree)."""
import json
import math
import sys

sys.path.insert(0, "src")
from trellis2.trellis3_r214 import WordTree  # noqa: E402
from trellis2.trellis3_r219 import WordTree219  # noqa: E402

ref, out, feats, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
S = json.load(open(ref, encoding="utf-8"))["sentences"]
T = WordTree(S, seed, feats) if feats == "all" or feats.startswith("func:") else WordTree219(S, seed, feats)
names = {str(b): T.names(b=math.inf if b == "leaf" else b) for b in [0, 1, 2, 3, 4, 5, 6, 6.5, 7, 7.25, 7.5, 7.75, "leaf"]}


def size_names(m):
    d = {}
    for w in T.words:
        node = T.leaf[w]
        while node.parent is not None and node.count < m:
            node = node.parent
        d[w] = "k%d" % node.id
    return d


for m in (2, 3, 5, 8, 12, 20):
    names["m%d" % m] = size_names(m)
json.dump({"seed": seed, "n": len(S), "feats": feats, "sentences": S, "names": names}, open(out, "w"))
print(feats, "types", len(T.words), {k: len(set(v.values())) for k, v in names.items() if k.startswith("m") or k in ("5", "6", "7")})
