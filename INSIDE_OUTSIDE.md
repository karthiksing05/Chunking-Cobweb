# Pivot to Inside Outside Parsing (branch `inside-outside`)

Our most recent revelation is that *inside-outside* parsing is deeply necessary to the goal of TRELLIS v2! My hope is that an inside-outside-inspired parsing scheme (and the implications of creating such an implementation) will set us on a path to success!

## What exactly does "inside-outside" parsing entail?

**Inside-outside** parsing is a form of parsing that extends beyond greedy parsing and considers multiple parsing probabilities at once - in fact, traditional inside-outside parsing considers every set of possible parses, computing the likelihood of each parse and "freezing" / selecting the best one.

In TRELLIS v1, we adopt a greedy scheme, which is shown to have successful results for smaller grammars but doesn't account for the creation of new symbols with respect to the broad grammar, resulting in an easy goal for one-off symbols to be repeatedly generated!

Honestly, humans kind of adopt a mental understanding of the sentence based on reading it for which chunks aren't REALLY a 1:1 application - but the goal is to show that structure helps you understand purely what is necessary for the target of coherence in language, not necessarily the same as a higher-level understanding!

## What do I want to achieve with this new scheme?

List of things I want to address with inside-outside parsing is below! By the constraints of the above framework, here are the things I hope to weave into the final framework as a result of including inside-outside parsing. I mention these ideas here because they most likely have to be designed in conjunction with the pivot of parsing.

*   Chunk Context! This is a big one - chunk context is extremely hard to standardize in a parsing scheme that is inherently greedy, but hopefully the process of full-parsing will allow us to refine existing representations with context
*   Unsupervised learning! This is also huge - having whole parses allows us to set up thresholds in a way that's far easier. We can also do something where we maintain candidate parses in a frontier and then learn them once we can confirm that they're good enough!
    *   Goal here is a little more precise - want to create a globally optimal and minimally viable grammar, borrowing from information-theory principles to do so in an incremental way

## Enriching our representations

### Chunk Context

We're observing a need to enrich context with higher-level structure!!

### Attention in Cobweb

Generally, it's important that we enrich context with attention to maintain long-range dependencies and make better representations! Again, the Cobweb-LLM repository (linked above) has important notes here, but hopefully we can simply do something more naive.