"""
Sandbox!

Currently working on testing json!
"""
print()

from parse import LanguageChunkingParser

parser = LanguageChunkingParser.load_state("unittests/ltm_analysis/halfway_test")

print(parser.get_long_term_memory().root.concept_hash())