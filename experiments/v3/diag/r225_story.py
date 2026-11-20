"""Round 225 driver: t3_story.py with the learner trellis2.trellis3_r225.Trellis3R225 (concepts_off switch; chunks off = store none in the config), without editing t3_story.py.
Usage: r225_story.py <the same arguments as t3_story.py> 
t3_story.py is read, patched in memory at one anchor (must match exactly once) and executed as __main__."""
import json
import os
import sys

sys.path.insert(0, "src")
from trellis2.trellis3_r225 import Trellis3R225  # noqa: E402

SRC = os.path.join("experiments", "v3", "diag", "t3_story.py")


def patch(src):
    old = "    L = Trellis3(seed=seed, **cfg)\n"
    assert src.count(old) == 1, "anchor not unique"
    return src.replace(old, "    L = _R225(seed=seed, **cfg, **json.loads(os.environ.get('R225_KW', '{}')))\n")


if __name__ == "__main__":
    src = patch(open(SRC, encoding="utf-8").read())
    sys.argv = [SRC] + sys.argv[1:]
    exec(compile(src, SRC, "exec"), {"__name__": "__main__", "__file__": SRC, "_R225": Trellis3R225})
