"""Silver constituency spans (Stanza's English parser, run in the `parse` env) for the
held-out sentences: every constituent covering 2+ tokens except the whole sentence.
Usage: silver_parse.py <heldout json> <out json>"""
import json
import os
import sys

import stanza

os.environ.setdefault("STANZA_RESOURCES_DIR", os.path.expanduser("~/flash/stanza_resources"))
sents = json.load(open(sys.argv[1], encoding="utf-8"))
nlp = stanza.Pipeline("en", processors="tokenize,pos,constituency", tokenize_pretokenized=True, verbose=False)
doc = nlp([list(s) for s in sents])


def spans(tree):
    out, pos = set(), [0]

    def walk(t):
        if not t.children:                      # a word
            pos[0] += 1
            return
        i = pos[0]
        for c in t.children:
            walk(c)
        out.add((i, pos[0]))
    walk(tree)
    return out


res = []
for s, sent in zip(sents, doc.sentences):
    sp = spans(sent.constituency)
    n = len(s)
    res.append({"tokens": s, "spans": sorted([i, j] for i, j in sp if j - i >= 2 and (i, j) != (0, n)),
                "tree": str(sent.constituency)})
json.dump(res, open(sys.argv[2], "w", encoding="utf-8"))
print(len(res), res[0]["tree"])
