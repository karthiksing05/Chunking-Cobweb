"""Is the neighbour-type facet (lc/rc) ever non-blank under the ntn config?"""
import sys, json
sys.path.insert(0, "src")
from collections import Counter
from trellis2.trellis3 import Trellis3, Elem
B = json.loads('{"store":"posterior","children":false,"spine":false,"refresh_budget":300,"heights":false,"type_names":"sample","type_key":"concept","refresh":"relabel","naming":"bands","restructure":"sample","word_cc":false,"neighbour_types":true,"word_content":"track"}')
L = Trellis3(seed=13, **B)
S = [s.split() for s in ["the dog saw a cat", "a cat saw the dog", "the man saw a dog", "a dog liked the cat"] * 5]
for s in S:
    L.learn(s)
vals = Counter()
for r in L.records:
    for a in ("lc", "rc"):
        v = r.ctx.get(a)
        vals["blank" if v in ("-", None) or v == {"-": 1.0} else "named"] += 1
print("type_bag entries:", len(L.type_bag), "type_res entries:", len(L.type_res))
print("lc/rc facet values over stored records:", dict(vals))
print("example ctx:", L.records[3].ctx)
