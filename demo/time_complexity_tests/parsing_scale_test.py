"""
Parsing Scale Test - a test to see how the amount of time it takes to parse data is on large scales

The implementation behind this test is also really simple. Our goal is solely to track whether the
amount of time taken to parse randomly generated sentences changes at some rate with respect to the
order at which they are passed in as well!

Good - linear rate of growth is super chill, the wide error bars are probably due to complexity of
the sentence (if I had to guess).
"""

from util.cfg import generate, TEST_CORPUS1, TEST_GRAMMAR1
from parse import LanguageChunkingParser
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import time

doc_len = 1000
datapoints = []
document = []

for _ in range(doc_len):
    sentence = generate("S", TEST_GRAMMAR1)
    document.append(sentence)

# Setting up the parser
print() # flush output just in case

parser = LanguageChunkingParser(TEST_CORPUS1, merge_split=True)

for i, sentence in enumerate(document):
    start = time.time()
    parse_tree = parser.parse_input([sentence], end_behavior="converge", debug=False)[0]
    end = time.time()
    parser.add_parse_tree(parse_tree, debug=False)

    secs = end - start
    sent_length = len(sentence.split(" "))

    datapoints.append((i, sent_length, secs))
    print(f"Time taken to add sentence {i} to LTM: {secs} seconds.")

datapoints = np.array(datapoints)
x = datapoints[:, 0]
y = datapoints[:, 1]
z = datapoints[:, 2]

# --- 1. Remove outliers (based on z-scores for x and z) ---
xz = np.column_stack((x, z))
z_scores = np.abs(stats.zscore(xz))
mask = (z_scores < 3).all(axis=1)
x, y, z = x[mask], y[mask], z[mask]

# --- 2. Define the complexity-style model function ---
def complexity_func(x, a, b):
    return a * np.power(x, b)

# --- 3. Fit the model to the data ---
# To avoid issues with negative or zero x, filter them
valid = x > 0
x_fit_data, z_fit_data = x[valid], z[valid]

popt, pcov = curve_fit(complexity_func, x_fit_data, z_fit_data, maxfev=10000)
a, b = popt

# --- 4. Generate smooth fit line ---
x_fit = np.linspace(min(x_fit_data), max(x_fit_data), 300)
z_fit = complexity_func(x_fit, *popt)

# --- 5. Create the scatter plot ---
plt.figure(figsize=(10, 6))
sc = plt.scatter(x, z, c=y, cmap='plasma', s=40, alpha=0.85, edgecolors='k', linewidth=0.3)
plt.plot(x_fit, z_fit, color='red', linewidth=2.5, label=f'Fit: z = {a:.3e}·x^{b:.3f}')

plt.colorbar(sc, label='Sentence Length (words) (heatmap scale)')
plt.xlabel('Document index (i)')
plt.ylabel('Time (seconds)')
plt.title('Scatterplot of document index vs time taken to parse with sentence-length-based Heatmap')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

plt.savefig("demo/time_complexity_tests/parsing_scale.png", dpi=300, bbox_inches='tight')
plt.show()