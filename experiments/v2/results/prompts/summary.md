Prompts: the first 1–3 words of held-out sentences (211 seen as the beginning of a training sentence, 22 unseen), each completed 5 times. Grammar: `experiments/v2/results/stories/grammar.pkl`.

| Completions | Prompts | Of the training length | Real | Every word triple attested | New | New and every triple attested |
|---|---|---|---|---|---|---|
| TRELLIS v2, temperature 1 | seen | 98% | 64.0% | 78.4% | 38.5% | 16.8% |
| TRELLIS v2, temperature 1 | unseen | 92% | 10.9% | 42.6% | 100.0% | 42.6% |
| TRELLIS v2, temperature 0.5 | seen | 99% | 77.8% | 91.4% | 24.8% | 16.2% |
| TRELLIS v2, temperature 0.5 | unseen | 92% | 13.9% | 54.5% | 100.0% | 54.5% |
| word bigram | seen | 71% | 52.5% | 80.1% | 50.7% | 30.9% |
| word bigram | unseen | 65% | 12.5% | 61.1% | 100.0% | 61.1% |
| word trigram | seen | 91% | 80.9% | 100.0% | 22.8% | 22.8% |
| word trigram | unseen | 74% | 30.9% | 86.4% | 100.0% | 86.4% |

Bits of the held-out sentence's actual continuation, given its prompt:

| Model | Seen prompts | Unseen prompts |
|---|---|---|
| TRELLIS v2 | 5.07 | 7.46 |
| word bigram | 6.96 | 8.16 |
| word trigram | 6.35 | 10.26 |

Completions (the prompt, then | ; the analysis as drawn, open chunks of the scaffold in ⟨ ⟩):

    seen    she                → ⟨⟨⟨she⟩ | [did not]⟩ [like it]⟩
    seen    she                → ⟨⟨she⟩ | [felt [very sad]]⟩
    seen    she                → ⟨⟨she⟩ | [felt [very happy]]⟩
    seen    she                → ⟨⟨⟨she⟩ | saw⟩ [[her friend] tom]⟩
    seen    she                → ⟨⟨she⟩ | [was [very happy]]⟩
    seen    they               → ⟨⟨they⟩ | [had [a [fun day]]]⟩
    seen    they               → ⟨⟨⟨they⟩ | are⟩ happy⟩
    seen    they               → ⟨⟨⟨they⟩ | were⟩ happy⟩
    seen    they               → ⟨⟨⟨they⟩ | were⟩ [very happy]⟩
    seen    they               → ⟨⟨⟨they⟩ | were⟩ [very happy]⟩
    seen    i                  → ⟨⟨⟨i⟩ | [wanted [to help]]⟩ lily⟩
    seen    i                  → ⟨⟨⟨i⟩ | can⟩ [do it]⟩
    seen    i                  → ⟨⟨⟨i⟩ | like⟩ you⟩
    seen    i                  → ⟨⟨⟨i⟩ | [want [to play]]⟩ too⟩
    seen    i                  → ⟨⟨⟨i⟩ | want⟩ it⟩
    seen    lily               → ⟨⟨⟨lily⟩ | is⟩ [not happy]⟩
    seen    lily               → ⟨⟨lily⟩ | [[[and ben] are] friends]⟩
    seen    lily               → ⟨⟨lily⟩ | [was [so happy]]⟩
    seen    lily               → ⟨⟨⟨lily⟩ | asked⟩ sue⟩
    seen    lily               → ⟨⟨lily⟩ | [[[and tom] were] happy]⟩
    seen    and                → ⟨⟨and⟩ | [you [[his play] together]]⟩
    seen    and                → ⟨⟨and⟩ | [they did]⟩
    seen    and                → ⟨⟨and⟩ | [they did]⟩
    seen    and                → ⟨⟨and⟩ | [you [like it]]⟩
    seen    and                → ⟨⟨and⟩ | [have [played [together mom]]]⟩
    seen    sue                → ⟨⟨sue⟩ | [was [very happy]]⟩
    seen    sue                → ⟨⟨⟨sue⟩ | wanted⟩ [to [be together]]⟩
    seen    sue                → ⟨⟨sue⟩ | [was [very happy]]⟩
    seen    sue                → ⟨⟨sue⟩ | [was happy]⟩
    seen    sue                → ⟨⟨sue⟩ | [was happy]⟩
