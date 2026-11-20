"""Round 219: word-tree class dump without any TRELLIS learning (the tree only needs the stream). Cheap (CPU), env trellis or chunks.
  r219_wordtree.py <ref wordtree json (gives the sentences)> <out json> <feats e.g. fn:25+d2+lr> <seed>
The out json has the r214 wordtree format (sentences, names for b in 0..7 and leaf) so r219_oracle.py R219_TREES can read it."""
import json
import math
import sys

sys.path.insert(0, "src")
from trellis2.trellis3_r219 import WordTree219  # noqa: E402

ref, out, feats, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
S = json.load(open(ref, encoding="utf-8"))["sentences"]
T = WordTree219(S, seed, feats)
names = {str(b): T.names(b=math.inf if b == "leaf" else b) for b in [0, 1, 2, 3, 4, 5, 6, 6.5, 7, 7.25, 7.5, 7.75, "leaf"]}
json.dump({"seed": seed, "n": len(S), "feats": feats, "sentences": S, "names": names}, open(out, "w"))
print(feats, "types", len(T.words), "classes@5", len(set(names["5"].values())), "@6", len(set(names["6"].values())), "@7", len(set(names["7"].values())))
