import sys
import os
import faulthandler
faulthandler.enable()
sys.path.insert(0, 'src')
from parse_mh import TRELLIS

vocab = ['the', 'cat', 'sat', 'on', 'a', 'mat', 'dog', 'ran', 'in', 'park']

w2 = TRELLIS(vocab, context_length=2, alpha=0.01, threshold=0.0, chunk_context=True)
print('ready', flush=True)

sentences = [
    'the cat sat on the mat',
    'the dog ran in the park',
    'a cat ran on a mat',
]
for i, s in enumerate(sentences):
    print(f'\n=== Sentence {i+1}: "{s}" ===', flush=True)
    pt = w2.parse_sentence(s, learning=True, debug=False)
    print(f'  Done.', flush=True)
