"""
CFG derivation-tree utilities used by grammar_decoding_test.py's
Phase 2 (chunk decoding).

Tree representation
-------------------
A derivation tree is either:
  - a terminal string (a word from the grammar's leaves), or
  - a 2-tuple ``(label, children)`` where ``children`` is a list of
    terminal strings and/or sub-trees.

So "the dog chased" might look like:

    ("S", [
        ("NP", [("Det", ["the"]), ("N", ["dog"])]),
        ("VP", [("V", ["chased"])]),
    ])

Functions
---------
- ``derive_phrase_labels(grammar)``      → set of non-terminal labels.
- ``generate_with_tree(symbol, grammar)``→ derivation tree.
- ``tokens_of(tree)``                    → list of terminal tokens.
- ``extract_brackets(tree, start_idx)``  → (spans, end_idx) where
                                           spans is a list of
                                           (start, end, label) tuples
                                           (inclusive indices) for each
                                           non-terminal subtree.

AdjP recursion mirrors the geometric-decay schedule in
``util.cfg.generate`` so derivation trees produced here have the same
adjective-count distribution as sentences from the regular generator.
"""

import random


def derive_phrase_labels(grammar):
    """Return the set of non-terminal labels in *grammar*."""
    return set(grammar.keys())


def generate_with_tree(symbol, grammar, _adjp_depth: int = 0):
    """Generate a derivation tree rooted at *symbol*.

    Returns either a terminal string (when *symbol* is a terminal) or
    a ``(label, children)`` tuple where ``children`` is a list whose
    elements are either strings or sub-trees.
    """
    if symbol not in grammar:
        return symbol  # terminal — return raw string

    # Depth-aware AdjP recursion to match util.cfg.generate.
    if symbol == "AdjP":
        productions = grammar[symbol]
        recursive   = [p for p in productions if "AdjP" in p]
        terminating = [p for p in productions if "AdjP" not in p]
        non_empty_term = [p for p in terminating if p]
        if non_empty_term:
            terminating = non_empty_term
        p_continue = (1.0 / 3.0) ** _adjp_depth
        if recursive and (not terminating or random.random() < p_continue):
            production = random.choice(recursive)
            next_depth = _adjp_depth + 1
        elif terminating:
            production = random.choice(terminating)
            next_depth = _adjp_depth
        else:
            production = random.choice(productions)
            next_depth = _adjp_depth
        children = []
        for sym in production:
            if sym == "AdjP":
                children.append(generate_with_tree(sym, grammar, next_depth))
            else:
                children.append(generate_with_tree(sym, grammar))
        return (symbol, children)

    production = random.choice(grammar[symbol])
    children = [generate_with_tree(sym, grammar) for sym in production]
    return (symbol, children)


def tokens_of(tree):
    """Return the list of terminal tokens (left-to-right) under *tree*."""
    if isinstance(tree, str):
        return [tree]
    _label, children = tree
    out = []
    for c in children:
        out.extend(tokens_of(c))
    return out


def extract_brackets(tree, start: int = 0):
    """Walk *tree* and return ``(spans, end_index)``.

    ``spans`` is a list of ``(start_token_idx, end_token_idx, label)``
    tuples — one per non-terminal subtree, with inclusive token
    indices. ``end_index`` is the first token index AFTER this
    subtree's span (so callers can chain children).

    Empty subtrees (e.g. ``AdjP → []`` if any grammar still allows it)
    are skipped — they cover no tokens.
    """
    if isinstance(tree, str):
        # Terminal — no bracket; consumes one token.
        return [], start + 1

    label, children = tree
    spans = []
    pos = start
    for c in children:
        if isinstance(c, str):
            pos += 1
        else:
            child_spans, new_pos = extract_brackets(c, pos)
            spans.extend(child_spans)
            pos = new_pos
    end = pos - 1
    if end >= start:  # skip empty productions
        spans.append((start, end, label))
    return spans, pos
