# Multiple Hierarchies Theory!

Whereas before we hypothesized needing multiple hierarchies for multiple levels of granularity or to separate primitives and composites, we now rely on multiple hierarchies to properly separate *content* and *context*.

## Prior problems

*   There's not really a problem, it's just that context gets treated weirdly and I think that separating into two hierarchies will make analysis a little cleaner and strengthen our chunk creation
*   It's also going to be a LOT easier to program a heuristic that makes sense and is adaptable
*   We've also evaluated that tracking stability and complexity of instances is incredibly important, and I think that the two hierarchy-partition will encode metadata appropriately

*   *Bonus but fun!* - splitting into two hierarchies will almost guaranteed make the diffusion model analogy a lot easier.

## Current solution

*   TWO COBWEB HIERARCHIES!! This will basically solve everything and provide so many new levels of analysis! I'll detail the process below

*   The two hierarchies:
    *   One Cobweb hierarchy will consist strictly of content!
        *   This hierarchy will specifically contain the content elements required to build chunks (that is, left and right content). 
        *   So this hierarchy will only store composites and statistics to build composites, nothing more than that!
        *   Path information will be weighted from the leaf to the root, weighted in an information-style way (leaf gets)
    *   One Cobweb hierarchy will consist strictly of context!
        *   Context hierarchy stores complexity!!!
        *   Assumes that all elements stored in here are purely in terms of context and only one element
        *   We should store the node reference of the content hierarchy for generation!! (Hidden attribute though)
            *   This will either be the basic-level node or the leaf-node itself - leaf node makes a lot more sense because we can just find the basic level of that leaf node and sample from it!!

*   Scoring:
    *   *Stability* - whether a chunk is built of two chunks that have been recognized prior over various occasions
        *   This should do with the complexity data that we annotate each chunk instance with
    *   *Recognition* - whether an instance of data is recognized by our memory
        *   From the cobweb-psych experiments, log-likelihood with respect to the whole tree (multinode expansion) yields success here!
        *   Only thing is I'm not sure whether 
    *   *Reusability* - whether we are building a chunk that we have the capability to reuse in the future
        *   This is important because we shouldn't build a ton of one-off chunks
        *   Most likely, this is a counts-focused heuristic, and will probably get better as we split into two hierarchies
    *   There were initial notes of whether the chunk "fit" here, but we don't need to worry about that as much - the idea is that recognition promotes goodness of fit anyways, so a separate mechanism isn't necessary


*   Learning: [*Given a sentence, our goal is to determine (unsupervised) a successful (partial or complete) parse for that sentence.*]
    *   **OPTIONAL:** We add primitive elements (as evidenced by )
        *   Messed up by either including or not including this, need to implement it as a yes/no setting in the 
    *   We start by creating a frontier of candidate chunks
        *   We select the best candidate chunk. The best candidate chunk is characterized by the following qualities:
            *   For now we'll just say tree-wide log-probability constitutes the best log-probability - we may need some work regarding the counts somewhere but this is overall fine.
            *   We might want to consider scores of both hierarchies as opposed to just one
        *   We add this chunk
    *   At the end, all candidate chunks are added to the content-hierarchy. All frozen chunks are added to both the content and context hierarchies.
        *   The reason for this shift is that only the content-hierarchy is in charge of aggregating frequencies

*   Performance: [*Given a partially filled in text, our goal is to generate the full parse and generate the sentence as an extension.*]
    *   Our input is of the form "word1 word2 [mask] word3 word4 [mask] word5 [mask]..." and corresponds to a level of granularity we've seen before (sentences, paragraphs, etc)
    *   The process for this is extremely straightforward: we simply build a parse from the bottom-up, making sense of as much partial prediction as possible and then denoise and generate from the top down
        *   We build a parse assuming that there is at most one "[mask]" word between any two words - the goal is to obtain structure for the words that do exist first before filling in the rest of the structure
        *   For a given node, we find the best node in the context hierarchy, which either expands the node or freezes it with one context element
            *   For freezing the node, there should just be one option to do, which is filling in the node with its word
            *   For finding the best expansion, we sample from the basic-level node of the leaf node filled in
        *   We continue until all of the parse tree has been generated, and then flatten all sentences
    *   Another option is that we build as much a parse tree as we can, and if the parse tree's been built to one node, we extend by one node and predict the right content.
        *   We can recursively do this until we run out of good predictivity for the right content!! 
        *   This also solves our prompt problem as we can recursively build a model that uses the prior context and generated results as context-left
        *   This also offers good generalizability to autoregressive archetypes - we predict the next high-level symbol based on prior context and then denoise it out!!
    *   **IMPORTANT NOTE** - we need to find some method of multi-level context that explains all the contextual levels of the parse tree, so that this data can be leveraged properly
        *   Perhaps the answer here is that we encode the level of appropriate context at that level
        *   Perhaps another answer here is that we don't need context for generation? Unless it's offered as the prompt

## Implementation 

Quick notes:
*   Need to make sure that hidden nodes work ok (this will involve some refactoring with the drawer)
*   Both the parse tree and the language-chunking-parser will need to undergo some changes!

I'll list all implementation details below!
---

Primitive Nodes:
*   Variables
    *   parent
    *   context_instance (created upon inception of the node)
    *   label - the path information for the context instance of this node, weighted appropriately, for creating higher-level nodes
    *   complexity - 1 for all Primitive Nodes
*   Methods
    *   [STATIC] create_node(context_instance, label)
    *   get_context_instance -> returns a context-focused instance of the data
        *   This stores content and complexity = 0 in hidden attributes!
    

Composite Nodes:
*   Variables
    *   parent, children
    *   is_global_root_node - to understand
    *   context_instance (created upon inception of the node)
    *   content_instance (created upon inception of the node)
    *   label - the path information for the context instance of this node, weighted appropriately, for creating higher-level nodes
    *   complexity - max(children.complexity) + 1 (or can also do sum(children.complexity))
*   Methods
    *   [STATIC] create_global_root
    *   [STATIC] create_instances(left_node, right_node)
    *   [STATIC] create_node(context_instance, content_instance, label)
        *   Creates the node with all the necessary attributes
    *   get_context_instance() -> returns a context-focused instance of the data
        *   This stores content = characterized leaf node and complexity = level of the tree from which it was parsed in hidden attributes!
    *   get_content_instance() -> returns a content-focused instance of the data
        *   This should have no metadata

Finite Parse Tree:
*   Variables
    *   index_map (hashmap): index -> node
*   Methods
    *   initialize_primitives
        *   each primitive will have an "index" which we can use to reference it and these indexes will be called from 
    *   generate_candidate_chunk(index1, index2)
    *   add_candidate_chunk(index1, index2) -> gives you new index
    *   Visualize methods for visualizing the chunk
        *   I would like for nodes to be represented by circles, and clicking on them expands both the content-instance and the context-instance
    *   to_json(filename) and from_json(filename) methods

Rolling Parse Tree: TBD

Long Term Memory:
*   Variables
    *   context_hierarchy = CobwebDiscreteTree
    *   content_hierarchy = CobwebDiscreteTree
    *   corpus
*   Methods
    *   Helper methods:
        *   instance to id-instance
        *   id-instance to instance
    *   get_content_instance_statistics
        *   A bit like the current FiniteParseTree.score_function - gives all the statistics we'll need to do thresholding and score calculation
    *   get_context_instance_statistics
        *   Same as above, but for the context hierarchy

Webster: (Primary Class - we're going to do all logic and parsing in here, and every other class is simply going to be a data-class that stores necessary information and helper methods)
*   Variables
    *   Corpus, attribute size, 
    *   An instance of the Long Term Memory
*   Methods
    *   parse_sentence(sentence, threshold=None, new_vocab=True, learning=False)
        *   Creates a parse tree for the sentence.
            *   We are in charge of maintaining the frontier of candidate chunks here
        *   threshold determines the threshold at which chunks are added
        *   new_vocab determines if new vocab is in the sentence
        *   if learning == true, we add the sentence at the end
            *   We should already have the top level of candidate chunks
    *   evaluate_chunk(content_instance, context_instance) - a method that, given the content and context instances for a candidate chunk, returns the necessary scores for that thing
        *   This should just call methods from the long-term hierarchy
    *   generate_sentence(masked_sentence)
        *   masked_sentence is of the form 

## Things to take note of

*   Need to figure out a context-only weighting style that works (perhaps negative attributes, sorting only by context should yield some valuable results)
    *   Perhaps bag-of-words-style weighting resurfaces here! Honestly, probably not - adjective order ruins the placeholders here which sucks, and a sliding window feels more feasible especially in accordance with the other stuff
    *   VERBALIZED: the reason that bag-of-words is bad is because it's not enough to explain the discrete relation that we set in stone here

*   Remember that this requires complexity for the generation and chunk building processes, cool things that we can do with stability here!

*   It might be true that building chunks requires both an understanding of context and content - HOWEVER, if we define every content item purely by its context (which I think is super reasonable) every chunk combination is naturally a context-driven creation!

## **Bonus:** Parallels to Diffusion Models, BPE + Tokenization

**TL;DR - we can use diffusion models and the diffusion process specifically to discover partonomies!**

*   Our generation process is precisely the same as diffusion models - however, diffusion models learn noise and then unmask that noise gradually
    *   If we're able to create "noise" constructively, we can basically do the process of creating chunks in the training stage (and by extension, create a compositional latent space!!)
    *   This is basically choosing parts to noise with respect to their frequency???? Instead of noising random parts of the sentence
    *   We also need to extend a mechanism here that expands symbols or freezes them (HAVE A COMPLEXITY SITUATION)

*   The key is how we introduce noise - if we introduce noise in a structured way, unraveling this noise will induce compositionality
    *   Introducing noise based on frequency could result in some promising results!!
    *   This may also introduce a faster training regime, as we don't try to learn denoising from 0 to 100, we learn various levels of denoising granularity
    *   **BPE does this!!!** The biggest difference between us and BPE is that our method basically does BPE over generalizations of words in addition to just words, in order to produce a higher-level vocabulary

*   Important changes from traditional diffusion architecture
    *   We noise two tokens at the time (mimicking our chunking process) based on frequency, then we predict in reverse
    *   Basically the idea is that we scaffold and learn parses at the same time, and the "noise space" that we implement learns a distribution that can be decoded into partonomies

*   Important changes from BPE and current tokenization:
    *   When we learn a higher-level symbol, we need to learn the syntactic and semantic relevance of that symbol (i.e. it is a noun phrase but it is also a noun phrase for a red dog)
    *   These generalizations need to be unpacked anyways but there's huge promise here in the world of tokenization!! We can store higher-clarity symbols and unpack them in the way that Pat and I are doing!
    *   If we create an LLM that's capable of parsing higher-level symbols into lower-level symbols, it's probably going to generate much more consistently
    *   The idea is that we're building an LLM with an adaptive vocabulary and latent space!! (**SEE BELOW POINTS, THIS ONE GOES CRAZY FRFR**)
        *   ADAPTIVE VOCABULARY LEARNED THROUGH THE POINCARE EMBEDDINGS SYSTEM - we still build a discrete cobweb but classify things continuously and then build an extension to an LLM that learns similarities!!
        *   If this system ever succeeds at a single-hierarhcy 

*   An important note comes into play when discussing this is whether semantic information will be appropriately translated, and I think that diffusion models will fare better than our purely discrete counterpart in using the latent space to leverage both semantic and syntactic correlation (contextually related)

*   Another note here is that while our current process resembles diffusion LMs, the continuous parse tree is more analogous to autoregressive models!!