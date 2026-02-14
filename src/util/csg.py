"""
Utilities for simple context-sensitive grammars (CSG) — toy examples and a small
generator. The representation mirrors the style used in `cfg.py` but the
productions are context-sensitive: each production's left-hand side is a
sequence (tuple) of symbols to match in the current sentential form and may
contain surrounding context.

Grammar representation:
- Keys are tuples of symbols (the LHS pattern to match).
- Values are lists of RHS expansions, where each expansion is a list of symbols.

The provided `generate` function repeatedly finds occurrences of any LHS
pattern in the current sentential form and replaces one randomly chosen
occurrence with a randomly chosen RHS until no more productions apply or the
`max_steps` limit is reached.
"""

import random
from typing import List, Tuple, Dict

# A minimal set of example grammars (toy/context-sensitive style).
# Keys are tuples (LHS sequence). Values are lists of RHS sequences (lists).

# Grammar 1: a small context-sensitive-style grammar that produces
# grammatical English-like sentences (implemented with single-symbol LHS
# tuples so the generator works like a CFG but still uses tuple keys).
CSG_GRAMMAR1: Dict[Tuple[str, ...], List[List[str]]] = {
	("S",): [["NP", "VP"]],
	("NP",): [["Det", "AdjP", "N"], ["Det", "N"]],
	("AdjP",): [["Adj", "AdjP"], ["Adj"], []],
	("VP",): [["V", "NP"], ["V", "PP"], ["V"]],
	("PP",): [["P", "NP"]],
	("Det",): [["the"], ["a"]],
	("N",): [["cat"], ["dog"], ["man"], ["woman"], ["park"], ["telescope"]],
	("Adj",): [["big"], ["small"], ["red"], ["quick"], ["lazy"]],
	("V",): [["saw"], ["liked"], ["chased"], ["found"], ["admired"]],
	("P",): [["with"], ["in"], ["on"], ["under"]]
}

CSG_CORPUS1 = (
	sum(CSG_GRAMMAR1[("Det",)], []) +
	sum(CSG_GRAMMAR1[("N",)], []) +
	sum(CSG_GRAMMAR1[("Adj",)], []) +
	sum(CSG_GRAMMAR1[("V",)], []) +
	sum(CSG_GRAMMAR1[("P",)], [])
)


# Grammar 2: a small context-sensitive example that uses an explicit
# surrounding context to transform a name placeholder into a descriptive
# phrase when it appears inside a particular frame. This shows using
# multi-symbol LHS patterns.
CSG_GRAMMAR2: Dict[Tuple[str, ...], List[List[str]]] = {
	("S",): [["Greeting", "PREDICATE"]],
	("Greeting",): [["hello", "NAME"]],
	("NAME",): [["alice"], ["bob"]],
	# When NAME appears followed by a predicate about sleeping, expand to
	# include an adjective: context-sensitive pattern ("NAME", "PREDICATE").
	("NAME", "PREDICATE"): [["NAME", "who", "is", "tired", "PREDICATE"]],
	("PREDICATE",): [["sleeps"], ["sits", "quietly"]]
}

CSG_CORPUS2 = ["hello", "alice", "bob", "sleeps", "sits", "quietly", "who", "is", "tired"]


# Grammar 3: an example that demonstrates a relative clause insertion via
# context-sensitive replacement producing natural-sounding sentences.
CSG_GRAMMAR3: Dict[Tuple[str, ...], List[List[str]]] = {
	("S",): [["NP", "VP"]],
	("NP",): [["Det", "N"], ["Det", "N", "RelClause"]],
	("RelClause",): [["who", "V"]],
	("VP",): [["V", "NP"], ["V"]],
	("Det",): [["the"], ["a"]],
	("N",): [["teacher"], ["student"], ["dog"], ["cat"]],
	("V",): [["teaches"], ["helps"], ["follows"], ["chases"]]
}



def generate(start, grammar: Dict[Tuple[str, ...], List[List[str]]], max_steps: int = 1000) -> str:
	"""
	Generate a string from a context-sensitive grammar.

	Parameters
	- start: initial sentential form (symbol string or list/tuple of symbols).
	- grammar: mapping from tuple LHS patterns -> list of RHS expansions.
	- max_steps: safety limit on number of replacement steps.

	Returns a space-separated string of the final sentential form.
	"""
	if isinstance(start, (list, tuple)):
		current: List[str] = list(start)
	else:
		current = [start]

	# Keep a stable list of keys (tuples) while we operate.
	keys = list(grammar.keys())

	for _ in range(max_steps):
		matches: List[Tuple[int, Tuple[str, ...]]] = []

		# Find all occurrences of any LHS pattern in the current sentential form.
		for i in range(len(current)):
			for key in keys:
				klen = len(key)
				if i + klen <= len(current) and tuple(current[i:i + klen]) == key:
					matches.append((i, key))

		if not matches:
			break

		# Choose a random match and a random RHS for that LHS.
		i, key = random.choice(matches)
		rhs = random.choice(grammar[key])

		# Replace the matched slice with the RHS expansion.
		current[i:i + len(key)] = rhs

	return " ".join(current)


if __name__ == "__main__":
	# Small interactive sanity checks when run directly.
	random.seed(0)
	print("CSG_GRAMMAR1 examples:")
	for _ in range(5):
		print(generate("S", CSG_GRAMMAR1, max_steps=50))

	print("\nCSG_GRAMMAR2 examples:")
	for _ in range(5):
		print(generate("S", CSG_GRAMMAR2, max_steps=50))
    
	print("\nCSG_GRAMMAR3 examples:")
	for _ in range(5):
		print(generate("S", CSG_GRAMMAR3, max_steps=50))

