TinyStories, sentences of 3–5 words over the 100 most frequent words: 5000 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 26.1 |
| word 2-gram model | 12.8 |
| word 3-gram model | 12.0 |
| TRELLIS v2 | 11.0 |

Grammar: 54 categories, 73 chunk types; 100% of training sentences analysed as one tree (1.00 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 30.7% | 26.8% | 55.3% | 21.3% |
| real | 70.5% | 74.6% | 51.4% | 81.4% |
| new and real | 1.2% | 1.4% | 6.7% | 2.7% |
| 3–5 words long | 98.6% | 98.2% | 72.0% | 92.8% |
| real, among those of 3–5 words | 71.5% | 75.9% | 63.7% | 86.1% |
| new and real, among those of 3–5 words | 1.2% | 1.3% | 1.7% | 1.3% |
| word pairs in TinyStories | 96.5% | 97.3% | 100.0% | 100.0% |
| word triples in TinyStories | 86.4% | 87.9% | 84.2% | 100.0% |
| mean length | 4.0 | 4.0 | 4.0 | 4.0 |
| perceived with the analysis it was generated from | 99.3% | 99.6% | – | – |
| chunks inside sentences found in TinyStories | 92.7% | 93.2% | – | – |

Largest categories:

- S0 (5000): they are happy, they were very happy, tim was very happy, he was very happy, tim was so happy, he was so happy, she was very happy, they were happy
- S18 (2102): was very happy, was so happy, was sad, was happy, was very sad, felt sad, was not happy, looked sad
- S22 (1794): tim, he, she, lily, tom, sue, spot, max
- S44 (1541): fun, together, it, help, that, bird, friends, day
- S53 (1513): happy, sad, fun, tree, too, friend, mom, max
- S1 (1513): very happy, so happy, very sad, not happy, so sad, so fun, all happy, very very sad
- S52 (1512): very, so, not, all, happy, his, their, named
- S45 (1414): a, to, like, the, and, all, big, not
- S8 (1183): to play together, to help, like it, be friends, together all day, and had fun, a big tree, do it
- S43 (1180): was, felt, looked
- S25 (1144): she, he, it, i, tim, we, tom, lily
- S13 (826): they played, it was, we can, can we, i can, it is, you can, he saw

Most frequent chunks inside training analyses: *very happy* (777), *was very happy* (542), *so happy* (384), *they were* (377), *was so happy* (335), *was sad* (335), *was happy* (268), *they are* (260), *very sad* (222), *was very sad* (178), *to help* (161), *to play* (140), *the bird* (139), *not happy* (127), *can i* (108), *felt sad* (98), *have fun* (94), *wanted to help* (93), *the cat* (89), *the dog* (80)


The grammar's own sentences (analysis as generated):

    [[[they were] [very happy]] friends]
    [[it was] [their [a dog]]]  (new)
    [[it [wanted [to help]]] her]  (new)
    [[it is] [very happy]]  (new)
    [[they played] [and [had fun]]]
    [lily [felt [very happy]]]  (new)
    [tim [was [so happy]]]
    [tim [was sad]]
    [tim [[[and sam] are] friends]]
    [tim [[[and sue] were] sad]]
    [[[they were] [very happy]] friends]
    [[[can we] play] together]
    [she [was [very happy]]]
    [[they are] happy]
    [[they all] [liked [it together]]]  (new)
    [sue [was [very sad]]]
    [tim [was [very happy]]]
    [tim [was [very happy]]]
    [they [have [and up]]]  (new)
    [[he said] back]
    [[he [wanted [to help]]] her]
    [[she [wanted [to help]]] sue]  (new)
    [[the boy] [you [like it]]]  (new)
    [asked [the [little it]]]  (new)
    [do [you [like that]]]  (new)

All samples, including partial analyses (pieces joined by ·):

    [[they are] happy]
    [[we are] happy]
    [[they were] happy]
    [[the boy] [loved [[to play] together]]]  (new)
    [she [was [very happy]]]
    [he [felt happy]]
    [tim [was [very happy]]]
    [tim [felt sad]]
    [[it was] [a cat]]
    [[he loved] [his there]]  (new)
    [tim [[[and sam] are] friends]]
    [[we did] it]
    [[they wanted] [to help]]
    [they [had fun]]
    [[they are] happy]

Held-out sentences (minimum-risk analysis):

    [she [was [very happy]]]
    [they [liked [it too]]]
    [[they do] [not [like tom]]]
    [[i have] [a [new toy]]]
    [[they are] [not [a toy]]]
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
