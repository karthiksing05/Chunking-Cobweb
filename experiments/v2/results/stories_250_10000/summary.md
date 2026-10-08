TinyStories, sentences of 3–8 words over the 250 most frequent words: 10000 learned from sentences alone, 500 held out (seed 13).

| Model | Held-out bits per sentence |
|---|---|
| word 1-gram model | 40.7 |
| word 2-gram model | 23.9 |
| word 3-gram model | 24.2 |
| TRELLIS v2 | 20.9 |

Grammar: 65 categories, 163 chunk types; 100% of training sentences analysed as one tree (1.00 top-level chunks per sentence).

| Generated sentences (1,000) | TRELLIS v2: its own sentences (whole trees) | TRELLIS v2: all samples | word 2-gram | word 3-gram |
|---|---|---|---|---|
| new | 68.7% | 68.1% | 86.8% | 63.0% |
| real | 35.3% | 35.8% | 22.2% | 46.8% |
| new and real | 4.0% | 3.9% | 9.0% | 9.8% |
| 3–8 words long | 94.9% | 95.4% | 68.9% | 86.6% |
| real, among those of 3–8 words | 37.1% | 37.4% | 24.4% | 52.4% |
| new and real, among those of 3–8 words | 4.1% | 4.0% | 5.2% | 9.7% |
| word pairs in TinyStories | 94.1% | 94.1% | 100.0% | 100.0% |
| word triples in TinyStories | 79.0% | 79.4% | 77.5% | 100.0% |
| mean length | 5.4 | 5.3 | 5.4 | 5.5 |
| perceived with the analysis it was generated from | 98.7% | 98.5% | – | – |
| chunks inside sentences found in TinyStories | 79.0% | 78.8% | – | – |

Largest categories:

- S13 (10715): the, and, to, an, it, in, happy, a
- S0 (10000): but then … happened, then something unexpected happened, they had … fun, they played … day, they are happy, do you … me, they were very happy, tim was very happy
- S46 (6076): friends, idea, fun, together, day, it, you, ball
- S16 (5805): happy, an idea, sad, very happy, friends, fun, scared, it
- S22 (5805): they were, they are, they had, you are, they played together, one day, thank you, it is
- S1 (4174): an idea, the park, help you, all day, fun together, in the park, and proud, and happy
- S10 (4049): a, and, you, to, the, so, they, he
- S33 (3671): was, had, big, saw, of, happy, help, to
- S3 (2459): the park, an idea, a big tree, and smiled, the bird, all day, of friends, for you
- S39 (2220): said, had, is, felt, saw, looked, asked, day
- S38 (2144): were, are, had, became, have, played, looked, felt
- S41 (1878): he, she, tim, it, one, lily, tom, thank

Most frequent chunks inside training analyses: *very happy* (395), *they were* (364), *wanted to* (362), *tim was* (310), *something unexpected* (310), *play with* (306), *something unexpected happened* (305), *it was* (259), *a big* (245), *an idea* (241), *he was* (239), *so happy* (234), *the bird* (220), *thank you* (218), *the cat* (215), *want to* (213), *they are* (190), *did not* (189), *she was* (182), *they had* (179)


The grammar's own sentences (analysis as generated):

    [bear [so too]]  (new)
    [[they were] [happy [and [[said [thank you]] [mom happy]]]]]  (new)
    [[max [[liked to] [play with]]] [his friends]]  (new)
    [[they were] [very happy]]
    [[lily said] [look [mom [a ball]]]]  (new)
    [[sam was] [shiny [and scared]]]  (new)
    [[[they [[like to] [play with]]] can] [[i be] fun]]  (new)
    [[he ran] [out [playing together]]]  (new)
    [[he was] [very [happy [ever fun]]]]  (new)
    [[it was] [[a big] dad]]  (new)
    [[you are] [my friend]]
    [[you can] [have [as animals]]]  (new)
    [[[the cat] ran] [to help]]  (new)
    [[tim asked] [[his mom] said]]  (new)
    [[[they all] played] together]
    [[he was] scared]
    [[[[sam and] [the dog]] are] happy]  (new)
    [[he looked] [at [[[tim [play with]] was] [so [surprised is]]]]]  (new)
    [then [[something unexpected] happened]]
    [then [[[a big] red] me]]  (new)
    [[they were] [[so happy] [[to see] it]]]
    [[they were] [very happy]]
    [[he saw] [[[a big] red] ball]]
    [they [[[want to] play] [[and have] fun]]]  (new)
    [[[[lily and] ben] [says yes]] [i do]]  (new)

All samples, including partial analyses (pieces joined by ·):

    [[lucy went] [to help]]  (new)
    [[[[tom and] lily] are] friends]
    [what [is [good [for you]]]]  (new)
    [[max was] [surprised [but happy]]]  (new)
    [but [then [[something unexpected] happened]]]
    [[[they all] laughed] [[and walked] away]]  (new)
    [yes [[let's play] [again soon]]]  (new)
    [[[[lily and] max] became] [best do]]  (new)
    [[they [[feel to] be]] pretty]  (new)
    [what [can we]]  (new)
    [[they had] [[so much] [[i be] scared]]]  (new)
    [[they [played together]] [[all day] long]]
    [i [will too]]  (new)
    [[he [did not]] [like [the idea]]]  (new)
    [then [[tim had] this]]  (new)

Held-out sentences (minimum-risk analysis):

    [[she was] [playing [with him]]]
    [but [then [[something unexpected] happened]]]
    [can [[you help] [me [find it]]]]
    [[[[amy [said to]] tom] look] [at [my toy]]]
    [[he [[wanted to] know]] [[what was] [in it]]]
    [[they [played together]] [[all day] long]]
    [[she thought] [[it was] [too hard]]]
    [[tom liked] [the idea]]
    [[tim was] [so happy]]
    [[[they [laughed and]] had] [[lots of] fun]]
    [soon [[he was] [home [and [[he went] inside]]]]]
    [but [then [[something unexpected] happened]]]
    [but [then [[he found] [[a big] box]]]]
    [i [am [[a nice] dog]]]
    [[it [[wanted to] be]] [friends [with bob]]]
