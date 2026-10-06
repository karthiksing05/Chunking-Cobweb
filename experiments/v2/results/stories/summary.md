TinyStories, sentences of 3–5 words over the 100 most frequent words: 5000 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 26.1 |
| word 2-gram model | 12.8 |
| word 3-gram model | 12.0 |
| TRELLIS v2 | 11.4 |

Grammar: 54 categories, 70 chunk types; 66% of training sentences analysed as one tree (1.49 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 8.9% | 32.5% | 55.5% | 21.1% |
| real | 92.2% | 69.6% | 51.2% | 81.4% |
| new and real | 1.1% | 2.1% | 6.7% | 2.5% |
| 3–5 words long | 98.9% | 90.6% | 71.7% | 92.6% |
| real, among those of 3–5 words | 93.1% | 75.6% | 63.6% | 86.5% |
| new and real, among those of 3–5 words | 1.0% | 1.1% | 1.5% | 1.3% |
| word pairs in TinyStories | 98.9% | 97.6% | 100.0% | 100.0% |
| word triples in TinyStories | 95.5% | 85.0% | 83.9% | 100.0% |
| mean length | 3.8 | 4.0 | 4.0 | 4.0 |
| perceived with the analysis it was generated from | 99.6% | 98.3% | – | – |
| chunks inside sentences found in TinyStories | 98.7% | 98.4% | – | – |

Largest categories:

- S9 (4171): you, too, a, what, do, that, it was, they played
- S10 (3291): they are happy, they were very happy, tim was very happy, he was very happy, tim was so happy, he was so happy, she was very happy, they were happy
- S1 (1844): tim, he, she, lily, tom, the bird, sue, the cat
- S3 (1829): was very happy, was so happy, was sad, was happy, was very sad, felt sad, was not happy, looked sad
- S22 (1571): they, he, she, it, tim, i, we, tom
- S4 (1489): very happy, so happy, very sad, not happy, to help, to play, not fun, very fun
- S19 (1486): happy, sad, help, play, fun
- S18 (1486): very, so, not, to
- S52 (1156): was, felt, wanted, looked, loved
- S24 (926): was, is, can, saw, played, mom, asked, said
- S29 (778): were, are, wanted, is
- S20 (778): it, together, bird, sad, dog, fun, mom, happy

Most frequent chunks inside training analyses: *very happy* (763), *was very happy* (534), *so happy* (384), *they were* (375), *was so happy* (332), *was sad* (327), *they are* (260), *was happy* (255), *very sad* (213), *was very sad* (176), *to help* (161), *to play* (140), *the bird* (135), *not happy* (128), *can i* (108), *felt sad* (102), *wanted to help* (94), *have fun* (88), *the cat* (80), *had fun* (76)


The grammar's own sentences (analysis as generated):

    [[they were] happy]
    [he [was [very happy]]]
    [he [was happy]]
    [lily [[[and ben] are] friends]]
    [[they were] happy]
    [lily [[[and tom] are] friends]]
    [he [felt sad]]
    [they [did it]]  (new)
    [lily [[[and tom] are] sad]]
    [he [was [very happy]]]
    [tim [was [so happy]]]
    [she [was [very happy]]]
    [[they were] happy]
    [they [have fun]]
    [ben [felt sad]]
    [he [was [very happy]]]
    [[the bird] [was happy]]
    [tim [was [very happy]]]
    [tom [was happy]]
    [they [have fun]]
    [[it we] [be friends]]  (new)
    [he [looked sad]]
    [he [was [so happy]]]
    [lily [was sad]]
    [lily [was [so happy]]]

All samples, including partial analyses (pieces joined by ·):

    [she [was [very happy]]]
    [[the dog] [was sad]]
    [she [felt happy]]
    [it is] · [not fun] · friend · tim  (new)
    [sue [was sad]]
    [they [felt sad]]
    [he [was [very happy]]]
    [he [was [very happy]]]
    [[the girl] [was [very sad]]]
    [[you are] happy] · friends · [was sad]  (new)
    [tim [was [very happy]]]
    [[tim is] happy]
    [[[can we] play] [and max]] · [to tom]  (new)
    [he [was sad]]
    but · [[it was] happy]  (new)

Held-out sentences (minimum-risk analysis):

    [she [was [very happy]]]
    [[they [liked it]] too]
    [[[they do] not] [like tom]]
    [[[[i have] a] new] toy]
    [[[they are] not] [a toy]]
    [lily [was [not happy]]]
    [[they are] friends]
    [she [was [so happy]]]
    [they [have fun]]
    [[they are] happy]
    [and [[they were] [very happy]]]
    [[they played] [together [all day]]]
    [they [have fun]]
    [sue [was sad]]
    [[they played] [all day]]
