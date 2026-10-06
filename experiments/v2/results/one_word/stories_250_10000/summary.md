TinyStories, sentences of 3–8 words over the 250 most frequent words: 10000 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 40.7 |
| word 2-gram model | 23.9 |
| word 3-gram model | 24.2 |
| TRELLIS v2 | 23.5 |

Grammar: 69 categories, 112 chunk types; 0% of training sentences analysed as one tree (3.22 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 98.5% | 88.4% | 86.3% | 61.7% |
| real | 22.0% | 17.7% | 23.1% | 46.3% |
| new and real | 20.5% | 6.1% | 9.4% | 8.0% |
| 3–8 words long | 13.9% | 79.6% | 67.5% | 88.8% |
| real, among those of 3–8 words | 12.2% | 19.8% | 25.3% | 50.7% |
| new and real, among those of 3–8 words | 1.4% | 5.3% | 5.0% | 7.5% |
| word pairs in TinyStories | 97.2% | 96.3% | 100.0% | 100.0% |
| word triples in TinyStories | 87.0% | 73.7% | 76.2% | 100.0% |
| mean length | 1.9 | 5.4 | 5.2 | 5.2 |
| perceived with the analysis it was generated from | 95.5% | 97.7% | – | – |
| chunks inside sentences found in TinyStories | 91.8% | 95.4% | – | – |

Largest categories:

- S68 (32218): i, fun, happy, friends, sad, tim, very happy, it
- S57 (2981): the, it, not, a, his, they, he, play
- S43 (2958): was, you, ball, bird, together, mom, said, cat
- S36 (2223): they, tim and sue, lily and ben, lily and tom, tim and sam, anna and ben, tom and lily, tom and mia
- S30 (1854): were, are, had, became, played, all, have, like
- S39 (1720): he, she, it, but, the, and, then, tim
- S28 (1629): was
- S37 (1627): tim, he, it, she, lily, tom, sue, the cat
- S15 (1501): and
- S31 (1330): said, is, had, saw, asked, says, felt, thought
- S58 (1155): to, you, in, and, said, it, little, they
- S4 (1137): play, help, be, go, see, make, eat, find

Most frequent chunks inside training analyses: *wanted to* (367), *something unexpected* (310), *play with* (299), *a big* (239), *an idea* (225), *something unexpected happened* (159), *liked to* (149), *played together* (143), *tim and* (133), *thank you* (128), *want to* (128), *loved to* (125), *did not* (120), *the park* (117), *lily and* (113), *tom and* (106), *know what* (88), *a lot* (87), *the cat* (71), *the bird* (71)


The grammar's own sentences (analysis as generated):

    can  (new)
    look  (new)
    as  (new)
    [it is]  (new)
    she  (new)
    inside  (new)
    tim  (new)
    i  (new)
    [his mom]  (new)
    be  (new)
    [his dad]  (new)
    [[they both] had]  (new)
    but  (new)
    [tim was]  (new)
    [he asked]  (new)
    [tom was]  (new)
    [they all]  (new)
    [[[lily and] [the bird]] [laughed and]]  (new)
    [he liked]  (new)
    [[[ben and] lily] looked]  (new)
    how  (new)
    [you are]  (new)
    [he gave]  (new)
    [he [[wanted to] [play with]]]  (new)
    [it was]  (new)

All samples, including partial analyses (pieces joined by ·):

    [tim was] · [very happy] · too  (new)
    [then sue] · knew  (new)
    the · toy · dog · came  (new)
    [[the dog] was] · [very happy] · [to play]  (new)
    i · [did not]  (new)
    [he ran] · [[to do] [something lot]]  (new)
    [he said] · no  (new)
    [the cat] · tim · thought · [[the cat] was]  (new)
    i · [play with]  (new)
    lucy · felt  (new)
    [[[tom and] tom] [[wanted to] [play with]]] · [his friend]  (new)
    i · have  (new)
    [but then] · [[a big] ball] · outside  (new)
    sue · [had [an idea]]
    [his friends] · laughed · together  (new)

Held-out sentences (minimum-risk analysis):

    [[[she was] playing] [with him]]
    [but [then [[something unexpected] happened]]]
    [[[can [you help]] me] [find it]]
    [[[[amy [said [to tom]]] look] at] [my toy]]
    [[he [[[wanted to] [know what]] was]] [in it]]
    [[they [played together]] [[all day] long]]
    [[[she thought] [it was]] [too hard]]
    [[tom liked] [the idea]]
    [[tim was] [so happy]]
    [[[they [laughed and]] had] [[lots of] fun]]
    [[soon [[he was] [home and]]] [[he went] inside]]
    [but [then [[something unexpected] happened]]]
    [[[but then] [he found]] [[a big] box]]
    [[i am] [[a nice] dog]]
    [[[it [[wanted to] be]] friends] [with bob]]
