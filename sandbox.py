#!/usr/bin/env python3
"""Sandbox: load a saved WEBSTER state, redistribute the content hierarchy,
and parse a random sentence.

Usage: python sandbox.py
"""
import os
import sys
import random

# Ensure local `src` is importable
HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from parse_mh import WEBSTER
from util.cfg import generate, TEST_GRAMMAR2


def main():
    save_dir = "unittests/gen_learn_test_mh/final_ltm_data"
    if not os.path.exists(save_dir):
        print(f"Saved WEBSTER state not found at '{save_dir}'. Run the gen_learn_test_mh to create it.")
        return

    print(f"Loading WEBSTER from: {save_dir}")
    w = WEBSTER.load_state(save_dir)
    print("Loaded WEBSTER; LTM vocab size:", len(w.ltm.id_to_value))

    # Generate a proper sentence from the test grammar and parse it before/after redistribution
    sent = generate("S", TEST_GRAMMAR2)
    print("Parsing sentence BEFORE redistribute:", sent)
    try:
        tree_before = w.parse_sentence(sent, threshold=None, new_vocab=False, learning=False, debug=True)
        if hasattr(tree_before, 'visualize'):
            tree_before.visualize("sandbox_parse_tree_before", render_png=False)
            print("Wrote sandbox_parse_tree_before.html")
    except Exception as e:
        print("Parsing before redistribute failed:", e)

    # Call redistribute on the content hierarchy (may raise if not implemented)
    try:
        print("Calling redistribute(2000) on content_hierarchy...")
        w.ltm.content_hierarchy.redistribute(2000)
        print("Redistribute completed.")
    except Exception as e:
        print("redistribute failed:", e)

    print("Parsing sentence AFTER redistribute:", sent)
    try:
        tree_after = w.parse_sentence(sent, threshold=None, new_vocab=False, learning=False, debug=True)
        if hasattr(tree_after, 'visualize'):
            tree_after.visualize("sandbox_parse_tree_after", render_png=False)
            print("Wrote sandbox_parse_tree_after.html")
    except Exception as e:
        print("Parsing after redistribute failed:", e)


main()
