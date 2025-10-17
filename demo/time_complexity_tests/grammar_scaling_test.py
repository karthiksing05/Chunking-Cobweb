"""
Grammar Scaling Test - A test to evaluate how parsing scales at different granularities of grammars.

CFGs can be measured in terms of two separate factors: the number of potential relations and the number
of items in the vocabulary. Additionally, while there does exist a "size" property for CFGs which sums
both rules and symbols involved in rules, it'll be important to denote the differences in experimental
time complexity between having a lot of symbols and having a lot of rules. So, we'll implement two 
separate tests, which either fix the length of the vocabulary or fix the number of rules, and track
changes in two separate graphs.

For this implementation, we'll need to aptly summarize rules and 
*   To poll subsets of the vocabulary, we can randomly sample from the vocabulary
    *   Create some fixed grammar similar to 
*   To poll subsets of the grammar rules, we'll have to include different parts of speech incrementally
    *   To do this I think we're just going to have GPT construct different grammars of varying sizes and
        complexities but the same vocabulary length
"""

from util.cfg import generate
from parse import LanguageChunkingParser
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
