"""Round 230 driver: r223_story.py (Trellis3R223: emission / gate level, context level, reach) on the controlled grammar of r230_grammar.py.
Only the data block of t3_story.py (the TinyStories reading and the gold tags) is replaced, by r230_grammar.load_corpus; nltk.pos_tag is the lexicon's tag.
Usage: r230_story.py <the same arguments as r223_story.py / t3_story.py>.  The learner, the dials and the generator are r223's, unchanged.
Env R214_WORDTREE=<out.json> (no learning): the word tree's class names at each level (r214's dump)."""
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r214_story as R  # noqa: E402
import r223_story as R3  # noqa: E402
import r230_grammar as G  # noqa: E402
from trellis2.trellis3_r223 import Trellis3R223  # noqa: E402

R.Trellis3R214 = Trellis3R223
if os.environ.get("R233_KW"):        # r233: the learner with Trellis3R233 options (parse_min_bits, chosen_mode); default unchanged
    import json as _json
    from trellis2.trellis3_r233 import Trellis3R233
    _KW = _json.loads(os.environ["R233_KW"])

    class _R233(Trellis3R233):
        def __init__(self, *a, **k):
            super().__init__(*a, **{**_KW, **k})
            if os.environ.get("R233_FORCE"):     # r233: learning-time parse forced to gold | rb (default off)
                import r233_goldparse as _GP
                self.force, self.force_gold = os.environ["R233_FORCE"], _GP.gold
    R.Trellis3R214 = _R233
    Trellis3R223 = _R233
A0 = "# stories as lists of sentences (as read_sentences splits them)"
A1 = "uni, pair, n_sent = Counter(), Counter(), 0"
INJECT = ("_G230 = _r230.load_corpus(seed, n)\n"
          "all_sents, vocab, keep, train, seen, val, lm_held, pairs, held, corpus, gold, patterns, by_tag, CONTENT = (_G230[k_] for k_ in "
          "('all_sents', 'vocab', 'keep', 'train', 'seen', 'val', 'lm_held', 'pairs', 'held', 'corpus', 'gold', 'patterns', 'by_tag', 'CONTENT'))\n"
          "k = 0\n")

if __name__ == "__main__":
    src = R.patch(open(R.SRC, encoding="utf-8").read())
    assert src.count(A0) == 1 and src.count(A1) == 1
    i0, i1 = src.index(A0), src.index(A1)
    src = src[:i0] + INJECT + src[i1:]
    B0 = "ret_c, _ = cross(P, ret)"
    assert src.count(B0) == 1
    src = src.replace(B0, B0 + "\nret_c = rand_c if ret_c is None else ret_c      # r230: sentences are independent, no stored next sentences")
    T0 = 'L.generate_tree(g_rng, index=ix, fill_name=FILL, **({"after": p} if cond else {}), **{"choose": "leaf", **arm})'
    if src.count(T0) == 1:    # r233: an arm may set fill_name (False = fill a part from its concept's counts alone); default unchanged
        src = src.replace(T0, 'L.generate_tree(g_rng, index=ix, fill_name=arm.get("fill_name", FILL), **({"after": p} if cond else {}), '
                              '**{"choose": "leaf", **{k_: v_ for k_, v_ in arm.items() if k_ != "fill_name"}})')
    G.install_nltk_stub()
    sys.argv = [R.SRC] + sys.argv[1:]
    g = {"__name__": "__main__", "__file__": R.SRC, "_R214": Trellis3R223, "_r214_start": R3._start, "_r214_end": R3._end,
         "_r214_levels": R._r214_levels, "_r214_wordmap": R._r214_wordmap, "_r214_make": R._r214_make, "_r230": G}
    exec(compile(src, R.SRC, "exec"), g)
