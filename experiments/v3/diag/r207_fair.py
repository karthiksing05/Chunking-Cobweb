"""Round 207: r204_fair.py with the escape arms ("escape": "n1", always with "edge": "exact") reported as family "TRELLIS edge w<N>"
(the window from the dump dir's name). Run it on the escape dirs only, plus the r200 frontier (LSTM / GPT), so the TRELLIS edge family IS the
escape arm; the plain edge arm's numbers are r204-fair-n*.perseed.tsv. Usage as r204_fair.py."""
import json
import re
import sys

import r204_fair as F

_orig = F.family_of


def family_of(r):
    arm = json.loads(r["arm"])
    if arm.get("escape") == "n1":
        m = re.search(r"-w(\d+)$", r["dir"])
        return "TRELLIS edge w%s" % m.group(1), arm.get("temp", 1.0)
    return _orig(r)


F.family_of = family_of
if __name__ == "__main__":
    F.main(int(sys.argv[1]), sys.argv[2], sys.argv[3:])
