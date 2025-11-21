"""
Thresholding Test - to govern and analyze the statistics by which chunks are aggregated by their content.
"""

from util.cfg import generate, TEST_CORPUS2, TEST_GRAMMAR2
from thresholding import ChunkThresholder
import numpy as np

num_sentences = 500
document = []

for _ in range(num_sentences):
    sentence = generate("S", TEST_GRAMMAR2)
    document.append(sentence)

id_to_value = TEST_CORPUS2
value_to_id = dict([(val, i) for i, val in enumerate(id_to_value)])

SPLIT_SIZE = 0.97
train_doc = document[:int(num_sentences * SPLIT_SIZE)]
test_doc = document[int(num_sentences * SPLIT_SIZE):]

thresholder = ChunkThresholder(id_to_value, value_to_id, strategy="pmi")

for sentence in train_doc:
    thresholder.aggregate(sentence)

k = 10
flat_idx = np.argpartition(thresholder.digram_frequency.ravel(), -k)[-k:]
rows, cols = np.unravel_index(flat_idx, thresholder.digram_frequency.shape)
print(f"Top-{k} Most Frequent Digrams (of {thresholder.n_digrams} total):")
for i in range(k):
    print(f"-    ({id_to_value[rows[i]]}, {id_to_value[cols[i]]}) = {thresholder.digram_frequency[rows[i]][cols[i]]}")

for sentence in test_doc:
    heuristics = thresholder.aggregate(sentence, update_stats=False)
    print(f"Sentence: {sentence}")
    sent_lst = sentence.split(" ")
    for i in range(len(sent_lst) - 1):
        print(f"-   Digram ({sent_lst[i]}, {sent_lst[i + 1]}) has heuristic {heuristics[i]}.")

    print("-" * 40)
    