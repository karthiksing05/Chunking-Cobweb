TinyStories, sentences of 3–5 words over the 100 most frequent words: 9000 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 26.1 |
| word 2-gram model | 12.7 |
| word 3-gram model | 11.5 |
| TRELLIS v2 | 10.7 |

Grammar: 71 categories, 103 chunk types; 100% of training sentences analysed as one tree (1.00 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 21.6% | 26.1% | 55.8% | 20.3% |
| real | 78.7% | 74.4% | 49.7% | 81.2% |
| new and real | 0.3% | 0.5% | 5.5% | 1.5% |
| 3–5 words long | 99.5% | 98.8% | 70.5% | 93.3% |
| real, among those of 3–5 words | 78.9% | 75.0% | 63.0% | 85.5% |
| new and real, among those of 3–5 words | 0.1% | 0.2% | 0.3% | 0.1% |
| word pairs in TinyStories | 97.5% | 97.2% | 100.0% | 100.0% |
| word triples in TinyStories | 90.6% | 88.6% | 85.3% | 100.0% |
| mean length | 4.0 | 4.0 | 3.9 | 3.9 |
| perceived with the analysis it was generated from | 99.3% | 99.7% | – | – |
| chunks inside sentences found in TinyStories | 95.9% | 95.2% | – | – |

Largest categories:

- S57 (9000): they are happy, they were very happy, tim was very happy, he was very happy, he was so happy, they were happy, tim was so happy, she was very happy
- S8 (3582): was, felt, looked, we, i
- S12 (3417): very happy, sad, so happy, happy, very sad, not happy, so sad, not sad
- S70 (3396): was very happy, was so happy, was sad, was happy, was very sad, felt sad, was not happy, looked sad
- S23 (2863): tim, he, she, lily, tom, sue, spot, max
- S0 (2627): very, so, not, all
- S9 (2626): happy, sad, friends, fun, big, little, there
- S2 (2042): to, a, like, the, not, and, all, be
- S13 (1974): it, fun, together, happy, friends, day, help, too
- S69 (1778): like it, be friends, not happy, together all day, a big tree, help you, to help, play together
- S24 (1764): they, we
- S60 (1677): it was, they played, can we, we can, do you, i like, i can, i want

Most frequent chunks inside training analyses: *very happy* (1384), *was very happy* (982), *so happy* (706), *they were* (658), *was so happy* (627), *was sad* (627), *was happy* (464), *they are* (455), *very sad* (379), *was very sad* (303), *wanted to* (284), *not happy* (247), *the bird* (241), *wanted to help* (199), *to play* (199), *have fun* (187), *can i* (183), *felt sad* (173), *it was* (166), *was not happy* (152)


The grammar's own sentences (analysis as generated):

    [[we are] friends]
    [[they looked] [and looked]]
    [she [was [not happy]]]
    [he [was sad]]
    [[they were] [very happy]]
    [lily [[[and tom] are] sad]]
    [she [was [so big]]]  (new)
    [[tom [[wanted to] help]] max]  (new)
    [lily [was [very happy]]]
    [she [was happy]]
    [[do you] [like it]]
    [he [was [so little]]]
    [he [was [very sad]]]
    [[they are] friends]
    [she [[wanted to] help]]
    [[they went] [up [and up]]]
    [he [was [so happy]]]
    [[they were] [all happy]]
    [they [have fun]]
    [[tom [was happy]] [to help]]
    [tim [was [so happy]]]
    [lily [[[and tom] were] happy]]
    [[they were] sad]
    [he [was [very happy]]]
    [they [liked [[to play] together]]]

All samples, including partial analyses (pieces joined by ·):

    [[can we] [play together]]
    [[we can] [all day]]  (new)
    [tom [was [very happy]]]
    [[can we] [be friends]]
    [it [was [very happy]]]
    [[they are] friends]
    [so [they did]]
    [[they are] [very sad]]
    [it [was [not sad]]]  (new)
    [sue [was sad]]
    [max [felt sad]]
    [tim [was [very happy]]]
    [[they are] [very happy]]
    [[sam looked] [to help]]  (new)
    [[the girl] [looked sad]]

Held-out sentences (minimum-risk analysis):

    [she [was [very happy]]]
    [[[they liked] it] too]
    [[they [do not]] [like tom]]
    [[i have] [a [new toy]]]
    [[they are] [[not a] toy]]
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
