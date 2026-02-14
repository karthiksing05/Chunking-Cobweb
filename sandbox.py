"""
Sandbox!

Double hierarchy implementation working (for now)!

Still some fixes but in the meantime, we need to test and verify that basic-level
nodes are working well!
"""

from util.cfg import generate, TEST_GRAMMAR1, TEST_CORPUS1
from parse_mh import WEBSTER
import shutil
import os
import random

DIRPATH = "unittests/gen_learn_test_mh"

webster = WEBSTER().load_state(DIRPATH)

webster.generate_sentence(debug=True)