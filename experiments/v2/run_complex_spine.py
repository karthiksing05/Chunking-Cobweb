"""run_complex.py with a different spine depth for the representation instances
(idea 3: less of where an element stands in its description, so a clause inside
an experience can share a category with a whole one).

    python experiments/v2/run_complex_spine.py --spine-depth 1 --grammar deep --train 2000 --seed 13
"""
import functools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "..", "src"), HERE]

import run_complex as RC  # noqa: E402


def main():
    argv = sys.argv[1:]
    depth = 2
    if "--spine-depth" in argv:
        i = argv.index("--spine-depth")
        depth = int(argv[i + 1])
        del argv[i:i + 2]
    RC.UnsupervisedLearner = functools.partial(RC.UnsupervisedLearner, spine_depth=depth)
    RC.Trellis2 = functools.partial(RC.Trellis2, spine_depth=depth)
    if "--out" not in argv:
        argv += ["--out", os.path.join(HERE, "results", f"complex_spine{depth}")]
    sys.argv = [sys.argv[0]] + argv
    RC.main()


if __name__ == "__main__":
    main()
