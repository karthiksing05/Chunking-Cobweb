TinyStories, sentences of 3–5 words over the 100 most frequent words: 2500 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 26.1 |
| word 2-gram model | 13.2 |
| word 3-gram model | 13.0 |
| TRELLIS v2 | 12.3 |

Grammar: 34 categories, 53 chunk types; 58% of training sentences analysed as one tree (1.66 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 12.6% | 42.5% | 57.7% | 23.6% |
| real | 89.6% | 60.9% | 48.8% | 80.9% |
| new and real | 2.2% | 3.4% | 6.5% | 4.5% |
| 3–5 words long | 98.4% | 87.9% | 71.6% | 92.2% |
| real, among those of 3–5 words | 90.2% | 68.1% | 63.3% | 86.6% |
| new and real, among those of 3–5 words | 1.4% | 2.7% | 4.2% | 3.7% |
| word pairs in TinyStories | 98.4% | 94.4% | 100.0% | 100.0% |
| word triples in TinyStories | 94.0% | 77.5% | 84.8% | 100.0% |
| mean length | 3.8 | 4.0 | 4.1 | 4.0 |
| perceived with the analysis it was generated from | 99.7% | 97.6% | – | – |
| chunks inside sentences found in TinyStories | 96.0% | 93.2% | – | – |

Largest categories:

- S12 (2696): it, you, friends, together, too, do, to play, is
- S9 (1446): they are happy, they were very happy, tim was very happy, he was very happy, they have fun, she was very happy, tim was so happy, he was so happy
- S10 (1444): happy, are, play, was, can, mom, help, dog
- S25 (1198): they, the, to, she, he, i, not, we
- S2 (982): tim, he, she, lily, tom, sue, the bird, max
- S13 (723): very, so, not, all
- S5 (722): very happy, so happy, very sad, not happy, all very happy, so sad, not sad, all so happy
- S14 (710): happy, sad
- S4 (703): tim was, they were, he was, she was, lily was, tom was, sue was, the bird was
- S3 (647): they are, tim was, they were, he was, she was, lily was, tom was, sue was
- S33 (641): happy, sad, friends, too, said, lily, sue, tom
- S27 (624): was, felt, is

Most frequent chunks inside training analyses: *very happy* (400), *tim was* (190), *they were* (189), *so happy* (187), *he was* (137), *they are* (129), *she was* (119), *very sad* (91), *lily was* (79), *sue was* (62), *tom was* (61), *have fun* (59), *to help* (57), *can i* (54), *wanted to help* (47), *the bird* (45), *can we* (40), *the bird was* (39), *not happy* (36), *had fun* (36)


The grammar's own sentences (analysis as generated):

    [[spot was] happy]
    [[she [wanted lily]] sad]  (new)
    [[the was] [not happy]]  (new)
    [[tom was] [not happy]]
    [[he was] sad]
    [they [have fun]]
    [[sue was] [very sad]]
    [[he is] happy]
    [[tom was] happy]
    [[sue was] sad]
    [[they were] happy]
    [[tim was] [so happy]]
    [[she was] sad]
    [[[the bird] was] [very happy]]
    [[[the bird] was] [so happy]]
    [[they were] happy]
    [[they were] happy]
    [[they were] [very happy]]
    [[they were] [very happy]]
    [[sue was] sad]
    [[they were] sad]
    [[she felt] sad]
    [[he was] [so happy]]
    [[tim was] [very happy]]
    [[it [wanted sam]] happy]  (new)

All samples, including partial analyses (pieces joined by ·):

    [[tim was] happy]
    [[tim was] happy]
    [[he was] [very happy]]
    [[tim [and sam]] are] · friends · is  (new)
    [[they are] happy]
    [[she was] [so happy]]
    would · you · like · it  (new)
    [[they are] [very happy]]
    tim · [had fun] · together  (new)
    [[they are] [very happy]]
    [[spot was] [very happy]]
    [he liked] · [tim too]
    [[can we] do] · that · what · do  (new)
    [she said] · [they loved]  (new)
    [[they were] happy]

Held-out sentences (minimum-risk analysis):

    [[she was] [very happy]]
    [[[they liked] it] too]
    [[they do] [[not like] tom]]
    [[[i [have a]] new] toy]
    [[they are] [[not a] toy]]
    [[lily was] [not happy]]
    [[they are] friends]
    [[she was] [so happy]]
    [they [have fun]]
    [[they are] happy]
    [and [[they were] [very happy]]]
    [[[they played] together] [all day]]
    [they [have fun]]
    [[sue was] sad]
    [[[they played] all] day]
