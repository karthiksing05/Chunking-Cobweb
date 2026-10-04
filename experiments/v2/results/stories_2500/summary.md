TinyStories, sentences of 3–5 words over the 100 most frequent words: 2500 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 26.1 |
| word 2-gram model | 13.2 |
| word 3-gram model | 13.0 |
| TRELLIS v2 | 17.1 |

Grammar: 31 categories, 43 chunk types; 62% of training sentences analysed as one tree (1.51 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2 | word 2-gram | word 3-gram |
|---|---|---|---|
| new | 78.3% | 57.6% | 22.8% |
| real | 30.0% | 48.3% | 81.6% |
| new and real | 8.3% | 5.9% | 4.4% |
| word pairs in TinyStories | 88.4% | 100.0% | 100.0% |
| word triples in TinyStories | 62.8% | 86.0% | 100.0% |
| mean length | 3.9 | 4.1 | 4.0 |
| perceived with the analysis it was generated from | 99.7% | – | – |
| chunks inside sentences found in TinyStories | 91.5% | – | – |

Largest categories:

- S0 (3766): it, they are happy, together, too, they were very happy, tim was very happy, you, they have fun
- S7 (1177): have fun, like, did, wanted to help, was, mom, had fun, is
- S18 (1130): they, the, a, she, you, he, to, what
- S23 (1026): was, is, felt, little girl was, big bird was, little boy was, little bird was, big cat was
- S22 (1026): tim, he, she, lily, tom, sue, the bird, the dog
- S1 (1025): tim was, he was, she was, lily was, tom was, sue was, the bird was, the dog was
- S17 (912): happy, sad, fun, tree, very sad, his
- S16 (796): very, so, not, all, a
- S5 (795): very happy, so happy, very sad, not happy, all very happy, so sad, a happy, all happy
- S25 (478): were, are, felt, all, saw, to be
- S19 (410): sad, happy, fun
- S27 (364): they, we, friends

Most frequent chunks inside training analyses: *very happy* (401), *tim was* (191), *they were* (191), *so happy* (190), *they are* (139), *he was* (137), *she was* (119), *very sad* (104), *lily was* (79), *have fun* (73), *not happy* (66), *tom was* (65), *sue was* (63), *to help* (61), *can i* (54), *wanted to help* (49), *the bird* (48), *had fun* (45), *can we* (40), *the bird was* (39)


Generated sentences (analysis as generated):

    not · so  (new)
    [[[the bird] was] [not happy]] · [[can fun] happy]  (new)
    [[she was] [very happy]]
    [[sam felt] happy] · [[tim was] happy] · [the there] · [the [had fun]]  (new)
    [[she are] friends]  (new)
    [they had]  (new)
    [[he was] happy] · [[they were] said] · [tim help]  (new)
    [they did] · [[tom was] sad]  (new)
    [what fun]  (new)
    [what [have fun]]  (new)
    day  (new)
    [max fun]  (new)
    [[[[little play] can] bird] cat] · [[lily play] said]  (new)
    said  (new)
    [a help]  (new)
    [[the can] said]  (new)
    tim  (new)
    [[spot was] sad]
    [[ben was] happy]  (new)
    [you not]  (new)
    [the fun]  (new)
    [it like]  (new)
    [big happy]  (new)
    it  (new)
    [[tim is] [very happy]]

Held-out sentences (minimum-risk analysis):

    [[she was] [very happy]]
    [[they liked] [it too]]
    [[they do] [[not like] tom]]
    [[i have] [[a new] toy]]
    [[[they are] not] [a toy]]
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
