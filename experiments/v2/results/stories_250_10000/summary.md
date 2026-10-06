TinyStories, sentences of 3–8 words over the 250 most frequent words: 10000 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 40.7 |
| word 2-gram model | 23.9 |
| word 3-gram model | 24.2 |
| TRELLIS v2 | 21.8 |

Grammar: 57 categories, 171 chunk types; 6% of training sentences analysed as one tree (3.10 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 25.9% | 78.7% | 84.0% | 62.1% |
| real | 75.3% | 26.6% | 27.2% | 46.5% |
| new and real | 1.2% | 5.3% | 11.2% | 8.6% |
| 3–8 words long | 100.0% | 83.6% | 67.3% | 86.9% |
| real, among those of 3–8 words | 75.3% | 30.0% | 28.4% | 51.6% |
| new and real, among those of 3–8 words | 1.2% | 4.5% | 4.6% | 7.9% |
| word pairs in TinyStories | 97.2% | 96.1% | 100.0% | 100.0% |
| word triples in TinyStories | 90.3% | 80.0% | 76.6% | 100.0% |
| mean length | 4.2 | 5.4 | 5.2 | 5.4 |
| perceived with the analysis it was generated from | 99.4% | 98.3% | – | – |
| chunks inside sentences found in TinyStories | 96.9% | 96.1% | – | – |

Largest categories:

- S11 (30412): i, fun, friends, happy, sad, very happy, it, they were
- S56 (7174): happy, was, day, you, ball, and, help, sad
- S33 (7102): very, and, the, to, so, a, you, it
- S34 (2885): he, she, tim, you, it, tom, lily, one
- S25 (2231): said, felt, is, mom, are, saw, day, asked
- S31 (1714): they
- S29 (1604): was
- S26 (1377): were, had, are, played, all, have, like, became
- S30 (1376): tim, he, it, she, lily, tom, sue, max
- S12 (1238): did not, played together, wanted to help, liked to play, smiled and, loved to play, said yes, said thank you
- S18 (934): tom, lily, ben, tim, sue, sam, max, mia
- S7 (878): a big, wanted to, know what, a lot, all day, a big red, a fun, a small

Most frequent chunks inside training analyses: *wanted to* (367), *play with* (314), *something unexpected* (306), *something unexpected happened* (288), *a big* (240), *an idea* (227), *had an idea* (203), *thank you* (183), *did not* (165), *then something unexpected happened* (153), *liked to* (149), *played together* (141), *want to* (126), *loved to* (125), *the park* (119), *wanted to help* (109), *lily and* (107), *tom and* (98), *tim and* (96), *know what* (85)


The grammar's own sentences (analysis as generated):

    [[[tom and] sam] sara]  (new)
    [lily [had [an idea]]]
    [[[the dog] lily] tom]  (new)
    [but [[something unexpected] happened]]
    [but [then [[something unexpected] happened]]]
    [then [[something unexpected] happened]]
    [then [had [an idea]]]  (new)
    [but [then [[something unexpected] happened]]]
    [tim [had [an idea]]]
    [[[ben and] lily] asked]
    [she [had [an idea]]]
    [[he was] it]  (new)
    [but [[something unexpected] happened]]
    [[they always] mom]  (new)
    [then [[something unexpected] happened]]
    [then [tim [had [an idea]]]]
    [then [[something unexpected] happened]]
    [mia [had [an idea]]]
    [but [then [[something unexpected] happened]]]
    [but [then [[something unexpected] happened]]]
    [tim [had [an idea]]]
    [then [she [had [an idea]]]]
    [and [they [played together]]]  (new)
    [but [[something unexpected] happened]]
    [they [had [an idea]]]

All samples, including partial analyses (pieces joined by ·):

    [[[tim and] lily] [smiled and]] · [said yes] · let's · [play together] · for  (new)
    [he [[wanted to] [play with]]] · me · that  (new)
    [you are] · so · fast
    [you can] · have  (new)
    [[[lily and] ben] were] · playing  (new)
    [and [[something unexpected] happened]]  (new)
    [they [did not]] · [[know what] to]  (new)
    [they played] · [and had]  (new)
    can · we · be  (new)
    [they played] · [and had]  (new)
    i'm · sorry · anna
    [they are] · happy · were  (new)
    [how [want to]] · play · together  (new)
    [they became] · good · friends
    [amy [[a to] play]] · [in [the park]]  (new)

Held-out sentences (minimum-risk analysis):

    [[[she was] playing] [with him]]
    [but [then [[something unexpected] happened]]]
    [[[can [you help]] me] [find it]]
    [[[[amy said] [to tom]] look] [[at my] toy]]
    [[[he [[wanted to] [know what]]] was] [in it]]
    [[they [played together]] [[all day] long]]
    [[[she thought] [[it was] too]] hard]
    [[tom liked] [the idea]]
    [[tim was] [so happy]]
    [[[[they [laughed and]] had] [lots of]] fun]
    [[soon [[[he was] home] [and [he went]]]] inside]
    [but [then [[something unexpected] happened]]]
    [[but then] [[he found] [[a big] box]]]
    [i [am [[a nice] dog]]]
    [[[it [[wanted to] be]] friends] [with bob]]
