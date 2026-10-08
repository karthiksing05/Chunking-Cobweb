Prompts: the first 1–3 words of held-out sentences (211 seen as the beginning of a training sentence, 22 unseen), each completed 5 times. Grammar: `experiments/v2/results/stories/grammar.pkl`.

| Completions | Prompts | Of the training length | Real | Every word triple attested | New | New and every triple attested |
|---|---|---|---|---|---|---|
| TRELLIS v2, temperature 1 | seen | 98% | 62.7% | 77.4% | 38.8% | 16.2% |
| TRELLIS v2, temperature 1 | unseen | 95% | 17.3% | 39.4% | 100.0% | 39.4% |
| TRELLIS v2, temperature 0.5 | seen | 98% | 77.0% | 91.4% | 25.5% | 16.9% |
| TRELLIS v2, temperature 0.5 | unseen | 90% | 22.2% | 55.6% | 100.0% | 55.6% |
| word bigram | seen | 70% | 52.6% | 81.2% | 52.0% | 33.2% |
| word bigram | unseen | 55% | 6.7% | 58.3% | 100.0% | 58.3% |
| word trigram | seen | 91% | 78.5% | 100.0% | 25.7% | 25.7% |
| word trigram | unseen | 76% | 33.3% | 85.7% | 100.0% | 85.7% |

Bits of the held-out sentence's actual continuation, given its prompt:

| Model | Seen prompts | Unseen prompts |
|---|---|---|
| TRELLIS v2 | 5.22 | 7.63 |
| word bigram | 6.96 | 8.16 |
| word trigram | 6.35 | 10.26 |

Completions (the prompt, then | ; the analysis as drawn, open chunks of the scaffold in ⟨ ⟩):

    seen    she                → ⟨⟨⟨she⟩ | [did not]⟩ [like it]⟩
    seen    she                → ⟨⟨she⟩ | [felt sad]⟩
    seen    she                → ⟨⟨⟨she⟩ | [did not]⟩ [like together]⟩
    seen    she                → ⟨⟨she⟩ | [was sad]⟩
    seen    she                → ⟨⟨she⟩ | [was [not happy]]⟩
    seen    they               → ⟨⟨⟨they⟩ | were⟩ [very happy]⟩
    seen    they               → ⟨⟨⟨they⟩ | are⟩ happy⟩
    seen    they               → ⟨⟨they⟩ | [liked that]⟩
    seen    they               → ⟨⟨they⟩ | [liked [the cat]]⟩
    seen    they               → ⟨⟨⟨they⟩ | played⟩ [and played]⟩
    seen    i                  → ⟨⟨⟨i⟩ | [and ben]⟩ said⟩
    seen    i                  → ⟨⟨⟨i⟩ | like⟩ [that [one too]]⟩
    seen    i                  → ⟨⟨⟨i⟩ | want⟩ tom⟩
    seen    i                  → ⟨⟨⟨i⟩ | can⟩ [do it]⟩
    seen    i                  → ⟨⟨⟨i⟩ | like⟩ it⟩
    seen    lily               → ⟨⟨lily⟩ | [was [very happy]]⟩
    seen    lily               → ⟨⟨lily⟩ | [[[and sam] are] friends]⟩
    seen    lily               → ⟨⟨lily⟩ | [[[and ben] are] friends]⟩
    seen    lily               → ⟨⟨lily⟩ | [[[and tom] were] happy]⟩
    seen    lily               → ⟨⟨lily⟩ | [[[and tom] are] happy]⟩
    seen    and                → ⟨⟨and⟩ | [[they were] it]⟩
    seen    and                → ⟨⟨and⟩ | [[they were] happy]⟩
    seen    and                → ⟨⟨and⟩ | [you [like that]]⟩
    seen    and                → ⟨⟨⟨and⟩ | [wanted [to help]]⟩ sue⟩
    seen    and                → ⟨⟨and⟩ | [[they were] [and have]]⟩
    seen    sue                → ⟨⟨sue⟩ | [was [so happy]]⟩
    seen    sue                → ⟨⟨sue⟩ | [was [so happy]]⟩
    seen    sue                → ⟨⟨sue⟩ | [was happy]⟩
    seen    sue                → ⟨⟨sue⟩ | [was [so happy]]⟩
    seen    sue                → ⟨⟨sue⟩ | [was sad]⟩
