"""Draw a larger corpus from a condition's generating grammar (data-size test).

Same format as the paper corpora (unique sentences, length-capped as the
originals); written to data/<corpus dir>_x<factor>.

Usage:
    python scripts/generate_bigger_corpus.py <condition> <n sentences> <max words> <seed>
"""
import os, sys, json, random, hashlib

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

from util.cfg import generate_with_merges
from trellis2.data import CONDITIONS, target_grammar

cond, n_target, max_words, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
out_dir = os.path.join(_HERE, "..", "data", f"{CONDITIONS[cond]}_x{n_target // 400}")
os.makedirs(out_dir, exist_ok=True)
grammar = target_grammar(cond)
random.seed(seed)
seen, attempts = set(), 0
while len(seen) < n_target and attempts < n_target * 200:
    attempts += 1
    text, merges = generate_with_merges("S", grammar, flatten_at_parent=("VP", "VPobj"))
    text = text.strip()
    if not text or text in seen or len(text.split()) > max_words:
        continue
    seen.add(text)
    with open(os.path.join(out_dir, f"cfg_{hashlib.md5(text.encode('utf-8')).hexdigest()[:8]}.json"), "w",
              encoding="utf-8") as f:
        json.dump({"sentence": text, "merges": merges}, f, indent=2)
print(out_dir, len(seen), "sentences", attempts, "attempts")
