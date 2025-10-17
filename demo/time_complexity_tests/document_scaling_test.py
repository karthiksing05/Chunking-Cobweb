"""
Document Scaling Test - A quick test to see how much time it takes to build an LTM
depending on the number of documents passed in.

Results: Evaluated experimentally that we get about O(N^2) for N documents - not sure how
significant that is in terms of other traditional parsing methods but it is very very cool!
"""

from util.cfg import generate, TEST_CORPUS2, TEST_GRAMMAR2
from parse import LanguageChunkingParser
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time

doc_lens = [
    1, 2, 5, 7,
    10, 20, 50, 70,
    100, 200, 500, 700,
    1000, 2000, 5000, 7000, 10000
]
datapoints = []
document = []

for _ in range(doc_lens[-1]):
    sentence = generate("S", TEST_GRAMMAR2)
    document.append(sentence)

# Setting up the parser
print() # flush output just in case
for doc_len in doc_lens:

    start = time.time()

    parser = LanguageChunkingParser(TEST_CORPUS2, merge_split=True)

    for i, sentence in enumerate(document[:doc_len]):
        parse_tree = parser.parse_input([sentence], end_behavior="converge", debug=False)[0]
        parser.add_parse_tree(parse_tree, debug=False)

    end = time.time()

    secs = end - start

    datapoints.append((doc_len, secs))

    print(f"Time taken to add {doc_len} sentences to LTM: {secs} seconds.")


x, y = np.array(datapoints).T

def complexity_func(x, a, b):
    return a * x**b

popt, pcov = curve_fit(complexity_func, x, y)

a, b = popt
print(f"Fitted function: y = {a} * x^{b:.3f}")
print(f"→ Approximate time complexity: O(n^{b:.3f})")

x_fit = np.logspace(np.log10(min(x)), np.log10(max(x)), 200)
y_fit = complexity_func(x_fit, *popt)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x_fit, y_fit, color='red', label=f'Fit: O(n^{b:.2f})')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Input size (n)')
plt.ylabel('Time (s)')
plt.title('Document Scaling Time Complexity Plot + Fit')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)

plt.savefig("demo/time_complexity_tests/document_scaling_test_log.png", dpi=300, bbox_inches='tight')

plt.show()