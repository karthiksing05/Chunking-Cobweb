TinyStories, sentences of 3–5 words over the 100 most frequent words: 5000 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 26.1 |
| word 2-gram model | 12.8 |
| word 3-gram model | 12.0 |
| TRELLIS v2 | 11.1 |

Grammar: 52 categories, 106 chunk types; 100% of training sentences analysed as one tree (1.00 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 29.4% | 28.1% | 55.5% | 21.5% |
| real | 72.8% | 73.3% | 51.2% | 81.2% |
| new and real | 2.2% | 1.4% | 6.7% | 2.7% |
| 3–5 words long | 98.4% | 97.8% | 72.3% | 93.0% |
| real, among those of 3–5 words | 73.8% | 74.5% | 63.3% | 85.8% |
| new and real, among those of 3–5 words | 2.0% | 1.0% | 1.8% | 1.4% |
| word pairs in TinyStories | 96.8% | 96.4% | 100.0% | 100.0% |
| word triples in TinyStories | 87.8% | 85.4% | 84.2% | 100.0% |
| mean length | 4.0 | 4.0 | 4.0 | 4.0 |
| perceived with the analysis it was generated from | 99.8% | 99.4% | – | – |
| chunks inside sentences found in TinyStories | 93.0% | 92.5% | – | – |

Largest categories:

- S0 (5000): they are happy, they were very happy, tim was very happy, he was very happy, tim was so happy, he was so happy, she was very happy, they were happy
- S10 (2097): was very happy, was so happy, was sad, was happy, was very sad, felt sad, was not happy, looked sad
- S26 (1868): was, felt, looked, very, happy, sad, not
- S9 (1867): very happy, sad, so happy, happy, very sad, not happy, so sad, his friend
- S33 (1788): tim, he, she, lily, tom, sue, spot, max
- S23 (1551): fun, it, together, that, help, bird, friends, day
- S21 (1519): very, so, not, his, their, named
- S20 (1502): happy, sad, friend, mom, max
- S28 (1328): a, to, like, the, and, big, all, not
- S8 (1205): to play together, to help, like it, be friends, together all day, and had fun, a big tree, do it
- S44 (1158): she, he, it, i, tim, we, tom, lily
- S38 (888): were, are, felt, dog, bird, make, play

Most frequent chunks inside training analyses: *very happy* (777), *was very happy* (542), *so happy* (384), *they were* (377), *was so happy* (335), *was sad* (334), *was happy* (269), *they are* (260), *very sad* (222), *was very sad* (178), *to help* (162), *to play* (140), *the bird* (139), *not happy* (128), *can i* (108), *felt sad* (98), *wanted to help* (93), *have fun* (89), *the cat* (89), *it was* (85)


The grammar's own sentences (analysis as generated):

    [but [[i have] there]]  (new)
    [tim [was [very happy]]]
    [what [did want]]  (new)
    [she [was sad]]
    [[[can i] help] you]
    [[it was] spot]
    [[the cat] [was sad]]
    [what [is that]]
    [[they were] [so happy]]
    [[tom asked] sue]
    [[tom is] [a dog]]
    [sue [was sad]]
    [[they all] [play together]]
    [she [was [very happy]]]
    [[she asked] [one day]]
    [they [had [a friend]]]  (new)
    [he [was sad]]
    [tim [was [very happy]]]
    [[he saw] [a [big bird]]]  (new)
    [[he is] [not happy]]
    [[they are] [very happy]]
    [once [have [not play]]]  (new)
    [[can we] [be friends]]
    [they [have [a toy]]]  (new)
    [that [was [a boy]]]  (new)

All samples, including partial analyses (pieces joined by ·):

    [tom [was sad]]
    [[they all] [had fun]]
    [[he wanted] [[to have] fun]]
    [tim [[[and sam] are] friends]]
    [[they are] happy]
    [[the boy] [was [very sad]]]
    [[they played] [together [all day]]]
    [tim [was happy]]
    [said [the bird]]
    [she [felt sad]]
    [[[can i] help] you]
    [[they were] [not happy]]
    [[they are] happy]
    [[he [wanted [to play]]] too]
    [[she [did not]] asked]  (new)

Held-out sentences (minimum-risk analysis):

    [she [was [very happy]]]
    [[[they liked] it] too]
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
