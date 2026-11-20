"""Round 223 driver: r214_story.py with the context-abstraction dials (Trellis3R223).

Usage: r223_story.py <the same arguments as t3_story.py / r214_story.py>   (repo root, compute node, env trellis)
Arm keys (besides r214's wreach / wemitk / pair ...): "cbits" (b_ctx: content words of the query window are read at level b of the word tree;
"leaf" / null = exact, 0 = wildcard) and "cfbits" (the same for the frequent / function words, default exact). Emission stays exact (no "wemit");
"wemitk" (K) defines the frequent / function set; the cfg must hold "wemit_tree": true and "wtree_feats" to build the word tree.
An arm without these keys behaves exactly as under r214_story.py."""
import math
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r214_story as R  # noqa: E402
from trellis2.trellis3_r223 import Trellis3R223  # noqa: E402

R.Trellis3R214 = Trellis3R223


def _b(v):
    return None if v in (None, "leaf", "inf") else float(v)


def _start(L, arm):
    R._r214_start(L, arm)
    L.ctx_bits = _b(arm.get("cbits"))
    L.ctx_func_bits = _b(arm.get("cfbits"))


def _end(L):
    R._r214_end(L)
    L.ctx_bits = L.ctx_func_bits = None


if __name__ == "__main__":
    src = R.patch(open(R.SRC, encoding="utf-8").read())
    sys.argv = [R.SRC] + sys.argv[1:]
    g = {"__name__": "__main__", "__file__": R.SRC, "_R214": Trellis3R223, "_r214_start": _start, "_r214_end": _end,
         "_r214_levels": R._r214_levels, "_r214_wordmap": R._r214_wordmap, "_r214_make": R._r214_make}
    exec(compile(src, R.SRC, "exec"), g)
