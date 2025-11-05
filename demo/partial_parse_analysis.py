"""
Partial Parsing Analysis!

Goal with this test is to make sure that partial parses are created optimally - that
is, the tree doesn't try to inspect additional meaning out of the parse when it doesn't
need anything. 

Need to find the correct metric to determine when the tree stops successfully parsing!

I'm going to start this off by playing around with the parse tree editor and seeing how
heuristics evolve over time, and then, hopefully, after a couple sample examples are fed in,
a resultant datastructure and base ltm can be parsed and saved.

The goal is to create a stable LTM for TEST_CORPUS1 and TEST_GRAMMAR1, in the hopes that we
can apply it to consistent behavior.
"""

from util.cfg import generate, TEST_CORPUS1, TEST_GRAMMAR1
from parse import LanguageChunkingParser
