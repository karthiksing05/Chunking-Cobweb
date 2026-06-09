# Soft vs. Hard Composition

Notes as a result of conversation from Cobweb meetings - not really sure where this is going but it's important to talk about for the sake of compositional learning!

Composition has two very different lens of being looked through. There is the relation-focused composition (i.e. a snowman is a "man made of snow") and the relation-abstracted composition (i.e. a Zamboni is like a lawnmower with the attribute of "to do with ice" mixed in). We refer to the former as a hard composition and the latter as a soft composition. Most neural net approaches focus on soft composition, blending the semantic ideas of two different things in a way that creates a thing that is a little bit of both. By contrast, the symbolic field and architectures within rely on this idea of discrete and unified relations as a method for explaining everything.

## Experiment: Replicating Zekun's "soft" composition

Zekun's original experiment leverages existing prototypes to craft a new distribution which can then be sampled from to yield exemplars of a new category. My primary goal was to revitalize this experiment using Cobweb, which stores distributions for prototypes and does not need mode finding.

However, Cobweb lacks a continuous landscape over which sampling can occur - the cool thing with a diffusion model is that you can “sample” an infinite amount of leaves once you reconstruct the distribution that is the query prototype. By contrast, we’re just reconstructing the original query in terms of our existing prototypes.

So, this Cobweb method of reconstructing unseen instances in terms of existing prototypes is more analogous to a representation learning scheme (fitting in with the bag-of-concepts idea used in TRELLIS v1). The idea is that we can represent unknown concepts and instances in terms of known concepts and the blend in which we represent them can be reverse-engineered to derive components!
*   The pro of using Cobweb in this way is that it's a direct analog to the "Cobweb-is-an-autoencoder" mechanic - the latents are recognized as the concepts we compose, and although the composition itself is soft, I'm sure there's a way to distill the discrete relations by summarize 

My eventual hope is that by implementing a scheme of soft composition within a symbolic model, picking concepts that can be composed to result in the original item, we can do the following:
*   Build systems that can leverage properties of extended composition
*   Discover the set of relations + primitives within our input data