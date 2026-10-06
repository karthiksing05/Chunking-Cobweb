TinyStories, sentences of 3–5 words over the 100 most frequent words: 2500 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 26.1 |
| word 2-gram model | 13.2 |
| word 3-gram model | 13.0 |
| TRELLIS v2 | 12.8 |

Grammar: 34 categories, 56 chunk types; 59% of training sentences analysed as one tree (1.65 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 15.3% | 48.3% | 57.8% | 23.5% |
| real | 89.0% | 54.9% | 48.6% | 80.9% |
| new and real | 4.3% | 3.2% | 6.4% | 4.4% |
| 3–5 words long | 98.9% | 86.9% | 71.1% | 92.4% |
| real, among those of 3–5 words | 89.7% | 62.9% | 63.6% | 86.5% |
| new and real, among those of 3–5 words | 4.0% | 3.5% | 4.2% | 3.7% |
| word pairs in TinyStories | 98.5% | 94.8% | 100.0% | 100.0% |
| word triples in TinyStories | 94.3% | 75.7% | 84.6% | 100.0% |
| mean length | 3.8 | 4.0 | 4.1 | 4.1 |
| perceived with the analysis it was generated from | 99.8% | 96.2% | – | – |
| chunks inside sentences found in TinyStories | 96.1% | 95.0% | – | – |

Largest categories:

- S9 (2656): it, you, friends, together, too, do, to play, is
- S0 (1466): they are happy, they were very happy, tim was very happy, he was very happy, they have fun, she was very happy, tim was so happy, he was so happy
- S26 (1463): was, happy, are, play, can, help, like, dog
- S23 (1445): they, the, to, she, he, i, not, we
- S32 (991): tim, he, she, lily, tom, sue, the bird, max
- S5 (739): very happy, so happy, very sad, not happy, all very happy, wanted to help, is happy, all happy
- S10 (739): very, so, not, all, wanted, is, felt
- S11 (729): happy, sad, to help, very sad, friends, fun
- S1 (719): tim was, they were, he was, she was, lily was, tom was, sue was, the bird was
- S4 (667): they are, they were, tim was, he was, she was, lily was, tom was, sue was
- S13 (659): happy, sad, friends, too, said, lily, sue, tom
- S29 (588): was, is, bird, are, cat

Most frequent chunks inside training analyses: *very happy* (386), *tim was* (191), *so happy* (186), *they were* (180), *they are* (137), *he was* (136), *she was* (119), *very sad* (94), *lily was* (79), *have fun* (66), *tom was* (62), *sue was* (62), *to help* (54), *can i* (54), *wanted to help* (46), *not happy* (45), *the bird* (45), *can we* (40), *the bird was* (39), *had fun* (38)


The grammar's own sentences (analysis as generated):

    [[[lily [and sam]] were] happy]  (new)
    [[he was] sad]
    [[they are] friends]
    [[tom was] [so happy]]
    [[he was] sad]
    [they [have fun]]
    [[sue was] [very sad]]
    [[they are] happy]
    [[tom was] happy]
    [[sue was] sad]
    [[tim was] sad]
    [[tim was] [so happy]]
    [[she was] sad]
    [[[the bird] was] [very happy]]
    [[[the bird] was] [not happy]]  (new)
    [[he was] happy]
    [[[the boy] was] sad]
    [[tim was] [very happy]]
    [[[the girl] was] sad]
    [[they are] happy]
    [[[the cat] was] [very happy]]
    [[she is] [very happy]]
    [[tim was] sad]
    [[she was] [very happy]]
    [[[the boy] was] [very happy]]

All samples, including partial analyses (pieces joined by ·):

    [[tom was] happy]
    [[[tim [and sue]] are] happy]  (new)
    [it was] · looked  (new)
    [[tim felt] sad]
    [i have] · at  (new)
    [[they are] [very happy]]
    [[she was] [so happy]]
    [[lily was] happy]
    [she felt] · [very happy]
    [[tim was] [very sad]]
    [[tim felt] sad]
    [they play] · together
    [[they are] sad]
    [the dog] · [[is sad] happy]  (new)
    [[sam was] happy]

Held-out sentences (minimum-risk analysis):

    [[she was] [very happy]]
    [[they liked] [it too]]
    [[they do] [[not like] tom]]
    [[[[i have] a] new] toy]
    [[[[they are] not] a] toy]
    [[lily was] [not happy]]
    [[they are] friends]
    [[she was] [so happy]]
    [they [have fun]]
    [[they are] happy]
    [[and [they were]] [very happy]]
    [[[[they played] together] all] day]
    [they [have fun]]
    [[sue was] sad]
    [[[they played] all] day]
