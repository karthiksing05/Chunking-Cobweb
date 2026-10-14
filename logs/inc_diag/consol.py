import os, sys, pickle
sys.path.insert(0, "/coc/flash10/ksingara3/Chunking-Cobweb/src")
sys.setrecursionlimit(100000)
from trellis2.unsupervised import UnsupervisedLearner
from trellis2.chart import Chart
L = pickle.load(open(sys.argv[1], "rb"))
U = UnsupervisedLearner(seed=L.seed)
for t in L.experiences:
    U.experiences.append(list(t))
trees = L.stored_trees()
model, bits = U._fit(trees)
print("learner state code", round(L.tables.bits(), 1), "| night consolidation of the same bracketings + labels:", round(bits, 1))
m2, b2, t2, steps = U._consolidate(trees)
print("  + viterbi re-analysis (night hard EM):", [round(b, 1) for b in steps])
