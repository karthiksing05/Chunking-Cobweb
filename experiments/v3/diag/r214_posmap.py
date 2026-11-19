"""Diagnostic only (POS is never used by the model): each word type's majority spaCy UPOS over the training stream of a wordtree dump.
Usage (env cobint): r214_posmap.py <wordtree.json> <out.json>"""
import json
import sys
from collections import Counter, defaultdict

import spacy
from spacy.tokens import Doc

D = json.load(open(sys.argv[1], encoding="utf-8"))
nlp = spacy.load("en_core_web_sm")
pos = defaultdict(Counter)
for s, d in zip(D["sentences"], nlp.pipe((Doc(nlp.vocab, words=s) for s in D["sentences"]), batch_size=64)):
    for w, t in zip(s, d):
        pos[w][t.pos_] += 1
json.dump({w: c.most_common(1)[0][0] for w, c in pos.items()}, open(sys.argv[2], "w"))
print(len(pos), "types", Counter(c.most_common(1)[0][0] for c in pos.values()).most_common())
