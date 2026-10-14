"""Trace the incremental learner's first characters: accepted moves, symbols, analyses.
Usage: python chars_trace.py SNAPSHOT_DIR N [config-json]"""
import json, os, sys
snap = sys.argv[1]
sys.path.insert(0, os.path.join(snap, "src"))
from trellis2.characters import default_ids_path, load_characters  # noqa
from trellis2.incremental import IncrementalLearner  # noqa

n = int(sys.argv[2])
cfg = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
chars, _ = load_characters(default_ids_path(), seed=13)
L = IncrementalLearner(seed=13, config=cfg)
orig = L._apply


def name(A):
    toks = sorted({k[1] for k in L.rec_by_sym.get(A, ()) if k[0] == "P"})
    return f"S{A}{{{' '.join(toks[:8])}{'…' if len(toks) > 8 else ''}}}"


def traced(move):
    if L._journal is None:
        kind = move[0]
        desc = kind
        if kind == "merge":
            desc += f" {name(move[1])} + {name(move[2])}"
        elif kind == "chunk" or kind == "compound":
            desc += " " + str(move[1])[:80]
        print(f"   apply {desc} bits {L.tables.bits():.1f}", flush=True)
    return orig(move)


L._apply = traced
for k, c in enumerate(chars[:n]):
    print(k, c.char, " ".join(c.tokens), flush=True)
    L.observe(c.tokens)
    print(f"  bits {L.tables.bits():.1f} K {L.tables.K} tops {L._top_syms(k)}", flush=True)
print("symbols:")
for A in sorted(L.members):
    print("  ", name(A))
