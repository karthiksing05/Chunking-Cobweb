# Soft vs. Hard Composition

Notes as a result of conversation from Cobweb meetings - not really sure where this is going but it's important to talk about for the sake of compositional learning!

Composition has two very different lens of being looked through. There is the relation-focused composition (i.e. a snowman is a "man made of snow") and the relation-abstracted composition (i.e. a Zamboni is like a lawnmower with the attribute of "to do with ice" mixed in). We refer to the former as a hard composition and the latter as a soft composition. Most neural net approaches focus on soft composition, blending the semantic ideas of two different things in a way that creates a thing that is a little bit of both. By contrast, the symbolic field and architectures within rely on this idea of discrete and unified relations as a method for explaining everything.

## Experiment: Replicating Zekun's "soft" composition within Cobweb

Zekun's original experiment leverages existing prototypes to craft a new distribution which can then be sampled from to yield exemplars of a new category. My primary goal was to recreate this experiment using Cobweb, which stores distributions for prototypes and does not need mode finding, to verify that Cobweb appropriately creates concepts that can be composed in a novel way and show principles of "soft" composition can still be generated in this symbolic framework.
*   The core principle in Zekun's approach, after he does mode-finding and generates a pool of concepts, is to select the top-K concepts by a score that measures the contributions of a given concept to the OOD query. 
    *   By contrast, we find concepts iteratively by running a BFS down the tree using the marginal gain score that Zekun proposes (rather than a pool). We expand 3% of nodes in the whole tree (an empirically found number) and select the best one at each interval to use for composition.
*   Our PoE formula is exactly the same as Zekun's - we compose in a pixelwise way, leveraging log-probs as the scores relative to the query (including a temperature, which we set to 0.1).
*   We measure various different metrics of Faithfulness and Generation from Zekun's paper.
    *   Faithfulness - how similar is the composition to the query image itself?
    *   Generalization - how similar is the composition to other images within the query's class?

The cool thing with a diffusion model is that you can “sample” an infinite amount of leaves once you reconstruct the distribution that is the query prototype. By contrast, we’re just reconstructing the original query in terms of our existing prototypes. So, this Cobweb method of reconstructing unseen instances in terms of existing prototypes is more analogous to a representation learning scheme (fitting in with the bag-of-concepts idea used in TRELLIS v1, but interpolating more than just a bag). Core idea is that we can represent unknown concepts and instances in terms of known concepts and the blend in which we represent them can be reverse-engineered to derive relations!
*   The pro of using Cobweb in this way is that it's a direct, interpretable analog to the "Cobweb-is-an-autoencoder" metaphor - the latent representation is the concepts we compose.
*   Another huge pro is that this method of composition automatically decides what parts are taken from each 
*   I'm sure there's a way to distill the discrete relations by summarizing common relations across a sample of objects, and very similarly, we can consider primitives by doing composite math along these relations!

Immediate results:
*   90% version is super cool - shows what exactly is missing at each interval and tries to represent within-dist queries through 1 concept
*   Blends of colors happen very frequently because the first concept is trying to explain as much of the image as possible - so you end up with a blend of colors to result in the foreground! You can imagine that if we utilize 

## Extension: Distilling primitives + discrete relations from SOFT composition!

This is one of my personal goals in an effort to bridge the gap between hard and soft composition - I wanted to see if we could find components of the pixels and how they were related by classifying and grouping and generalizing over the different regions of correlated components.

I implemented the following procedure:
*   We take each donation-heatmap from the prior section's 90%-variant (continue selecting concepts until 90% of the image has been explained) and add them all to a matrix M of size (observations, n_pixels) (for 90%-variant, it's ~1600 obs. x 1024 pixels)
*   Then, we compute the cross-correlation matrix for each pixel with respect to all other pixels, so we now have matrix R of size (n_pixels, n_pixels)
*   Each pixel is now an instance, with its instance description being a 1024-dim vector of correlations to all other pixels (pixel corr for itself is 1).
*   We sort these 1024 pixel instances in a SEPARATE cobweb tree! The idea is that through this, we can find sections of pixels with high correlations (essentially clustering the correlations to see if there are regions of relevance!)

Overall, very interesting results! The biggest thing is the rings of value and then also the key digit expressed being a combination of a circle with a stalk (circle def inherits from 4, 9, 8, 2, 3, 5 while stalk inherits from 1, 7, 9). An important note is separation of foreground from background.

## Tenets of Composition

As a result of the experiment above, we see that soft composition can be employed to merge different ideas within a concept

My eventual hope is that by implementing a scheme of hard AND soft composition within a symbolic model, we can do the following:
*   Build systems that can leverage properties of soft AND hard composition
*   Discover the set of relations + primitives within our input data as a result of soft composition
    *   The set of primitives and relations can (hopefully) both be inspected by noticing commonalities in existing compositions, and new primitives may be able to be hypothesized by seeing how breaking down an OOD example with an inferred set of relations!
*   Similar to a "neuro-symbolic pairing", identify a reasonable mapping from soft composition to hard composition (or justify that one is a subset of the other)

## Compositional Autoencoder??

Is there validity to designing an autoencoder that works off the basis of developing concepts given an input and then generating output using PoE?
*   Instead of backprop, the idea is to somehow train the concept map to minimize the number of concepts needed to generate the output (in the hopes that it results in eventual disjoint ideas)
*   More on this in the 