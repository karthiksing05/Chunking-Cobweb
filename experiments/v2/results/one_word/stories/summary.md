TinyStories, sentences of 3–5 words over the 100 most frequent words: 5000 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 26.1 |
| word 2-gram model | 12.8 |
| word 3-gram model | 12.0 |
| TRELLIS v2 | 11.9 |

Grammar: 52 categories, 63 chunk types; 64% of training sentences analysed as one tree (1.55 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 16.1% | 41.2% | 55.6% | 21.3% |
| real | 86.0% | 61.2% | 50.6% | 81.2% |
| new and real | 2.1% | 2.4% | 6.2% | 2.5% |
| 3–5 words long | 98.2% | 88.3% | 71.6% | 92.5% |
| real, among those of 3–5 words | 87.4% | 68.5% | 63.5% | 86.4% |
| new and real, among those of 3–5 words | 1.9% | 1.9% | 1.5% | 1.3% |
| word pairs in TinyStories | 98.3% | 96.3% | 100.0% | 100.0% |
| word triples in TinyStories | 92.7% | 80.7% | 83.9% | 100.0% |
| mean length | 3.8 | 4.0 | 4.0 | 4.0 |
| perceived with the analysis it was generated from | 99.4% | 98.7% | – | – |
| chunks inside sentences found in TinyStories | 97.6% | 95.6% | – | – |

Largest categories:

- S0 (4546): you, too, what, a, do, that, it was, and
- S8 (3215): they are happy, they were very happy, tim was very happy, he was very happy, tim was so happy, he was so happy, she was very happy, they were happy
- S24 (1616): was
- S18 (1616): was very happy, was so happy, was sad, was happy, was very sad, was not happy, was not sad
- S5 (1529): they, it, she, he, i, we, tim, tom
- S3 (1525): is, was, can, saw, with, played, mom, was happy
- S23 (1401): happy, sad, friends, not
- S22 (1401): very, so, not, did, all
- S20 (1399): very happy, so happy, very sad, not happy, not friends, all happy, not sad
- S15 (1384): tim, he, she, lily, tom, sue, spot, max
- S7 (724): the, a, not, her, had, very, his, play
- S16 (662): they, she, lily, and

Most frequent chunks inside training analyses: *very happy* (774), *was very happy* (535), *so happy* (384), *they were* (375), *was so happy* (332), *was sad* (327), *they are* (255), *was happy* (255), *very sad* (192), *was very sad* (176), *the bird* (134), *wanted to help* (116), *to help* (116), *can i* (108), *felt sad* (102), *to play* (102), *not happy* (92), *have fun* (80), *the cat* (79), *can we* (75)


The grammar's own sentences (analysis as generated):

    [[they were] happy]
    [she [was [very happy]]]
    [she [was sad]]
    [[he is] [wanted friends]]  (new)
    [[he is] happy]
    [she [was sad]]
    [he [was [very happy]]]
    [[they were] [very happy]]
    [they [have fun]]
    [[they were] happy]
    [[they are] happy]
    [lily [was [so happy]]]
    [[the bird] [wanted [to help]]]
    [she [was [very happy]]]
    [it [was [not happy]]]  (new)
    [they [liked [[to play] together]]]
    [tim [was sad]]
    [she [was [very sad]]]
    [it [was [so happy]]]
    [[the bird] [looked sad]]
    [she [felt happy]]
    [he [was [very happy]]]
    [[they were] [very happy]]
    [they [felt sad]]
    [tim [was [so happy]]]

All samples, including partial analyses (pieces joined by ·):

    [he was] · [so bird]  (new)
    [tom [was sad]]
    [[they are] sad]
    [she liked] · [the toy] · was  (new)
    and · [it do] · you · make · friends  (new)
    [[they are] happy]
    [[they were] happy] · too · [[very [[to play] together]] cat]  (new)
    [lily [was happy]]
    [she liked] · [tim said] · that · is  (new)
    [sue [was sad]]
    [tim [was [very happy]]]
    [sue [was [so happy]]]
    [he [was [very happy]]]
    [max [felt sad]]
    what · did  (new)

Held-out sentences (minimum-risk analysis):

    [she [was [very happy]]]
    [[[they liked] it] too]
    [[[they do] not] [like tom]]
    [[i have] [[a new] toy]]
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
