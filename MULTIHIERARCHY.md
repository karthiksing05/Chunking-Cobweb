# Multiple Hierarchies Theory!

Whereas before we hypothesized needing multiple hierarchies for multiple levels of granularity or to separate primitives and composites, we now rely on multiple hierarchies to properly separate *content* and *context*.

## Methodology 4.0

I think Pat's right - we need to store pointers and use path information in real time for maximal simplicity, if we're implementing the redistribute method we should store leaves and move from there.

TODO LIST FOR NEXT METHODOLOGY (WITH REDISTRIBUTE):
*   Adjust Cobweb to store IDs that have pointers to Cobweb nodes and then modify the log-prob functions to recompute path information in real time (this is either a cobweb library approach or a )
    *   As a result of this, we can remove all of the instance merging and removing, and recompute everything in real time which should be really nice
*   CHUNK CONTEXT HOWWWWW
    *   Need to program a test that uses the leaves and then finds similarities in a similar way to the above thing - I think chunk context will be significantly easier if we overhaul with the new thing that we're trying to do!!

Adjustments for Cobweb:
*   Let's build the new ID-mapping natively into this - then, based on whether the given ID links to a string or a CobwebNode, we can treat it separately
    *   Assume that a string and a CobwebNode have no similarity

## Methodology 3.1

Lots more work to do - I'm going to spend some time devising a TODO list of stuff that needs to happen, in this order, for us to complete the framework, and then we're basically just going to iterate through this todo list.

We did stuff! Long term, shuffling makes things weird, alphas customized for the basic level work, and more data was needed for success!!
*   Initial results from cramming for CoCo reveals some, but not all consistency!!
*   The framework is better than I thought - just need to find strong hierarchy quality for generation!
    *   We see some proper disambiguation, hierarchical structure, the primary problem is that path representation is weak as a result of the poor hierarchy quality!!
    *   This is also important in generation - basic level nodes need to appropriately sample the level of diversity that they represent for the ability to do strong generation diversity

*   IMPORTANT THING: We need a simplicity bias - or maybe just better levels of generalization and better hierarchy quality LOL, not sure which yet.
    *   The primary reason I say this is that we need some bias that convinces the model to build a chunk that it recognizes before trying to create a new chunk (so perhaps basic level categories are either too broad or too narrow right now)

Potential things to try:
*   Redistribution!!!! GET THIS WORKING, THIS HAS TO BE IT FOR SURE
    *   Do Pat's thing - as we traverse down the path, we look at leaves of every given node and evaluate them as misplaced
    *   Remove MERGING AND SPLITTING AND IF THIS IS TRUE, DO IT!!!
*   Threshold by both log-prob and counts if needed
*   Program sufficient generation scheme!!
*   IF WORST COMES TO WORST - we're DEFINITELY going to have to store models of various depths in various hierarchies and call them all the same hierarchy (make complexity a visible attribute first, and then try this???)
    *   Create a Cobweb hierarchy neural model where we have LAYERS of Cobweb hierarchies

## Methodology 3.0!!

The home stretch! Going to add some notes regarding next steps here, but literally the thing holding us back right now is the quality of the context-hierarchy. Long-story short, we need to improve performance of the two hierarchies and then we're set!

WE DID IT!!! Kind of - still definitely need to mess around with truncating hierarchies and ALPHA TUNING IS BIGGEST PROBLEMMMM (tentative rule in 1 / vocab and then 1 for basic level calls)

Important general realizations below:
*   An important note here is that the basic level should only determine a passage - after that, we use LOG PROB OF THE TREE IN CONTENT HIERARCHY TO GENERATE BEST MATCHES
*   Having sufficiently low alpha helps hierarchy quality immensely, but reduces the basic level threshold. What we really need to do is figure out the right settings for alpha and get_basic such that we derive a sufficient basic level while having a hierarchy that makes sense!!
    *   Need to design a test that shows poor alphas at high values and whatever (lowkey idk but we need to figure this out!!!)
    *   One thing I've added is a primitive threshold for the context hierarchy - maybe we can extend this to the two-part 
    *   *Trying a basic level alpha that is different from the tree alpha!!! So far, so good!!!*
*   Redistribution does a really solid job of making the hierarchies better - which is extremely interesting
    *   Redistribute simply just can't be updated because it's terrible for paths - this would be significantly better if labels were just represented by basic-level av-counts but unfortunately this is not the case!

Some new notes for generation:
*   We should do some probability-based structure inference for what doesn't exist, and then be able to fill in that structure through probability
    *   Evaluate 
*   We need to find a fix for if we truncate the hierarchy where we still store a joint distribution at the leaves!!

**NEW GENERATION LOOP**
*   Given a node, we should predict the most likely structure next to it (either left or right) and then from structure one at a time
*   Then, from the top-down, we should pick candidates that we've seen that validate the structure we've chosen

Here's some pseudocode for generation:


Still may need the following things to improve the hierarchy - currently, reducing alpha sufficiently gets that done but it has harmful effects on the basic level node!
*   Observation buffer is HUGE - need to include variance of response here so that we know which to add?
*   Truncating hierarchy could be optimal - I like this just generally in terms of keeping the basic-level strongly concentrated at the root as an option
*   Adjusting alphas + representations is pretty optimal

## Methodology 2.0

A whole host of new problems! I'm refactoring and appending problems below so that we can discuss them properly, but very very promising stuff!!

### FROM DISCUSSIONS WITH CHRIS!
*   Unseen things are being seen with the log probabilities in bad ways!!! (***)
*   Not a bug but log prob is contingent on how the tree is built, and also not enough to guarantee frequency - we need some basic level adaptation + counts
*   Basic level seems pretty strong in our test_logprob_paths script! So maybe the problem is truly shuffling the data!!
    *   We can say that if the basic level evaluates to being the root node, that we haven't formed a basic level yet!! As per test_logprob_paths!!

### Revising our scoring (with both Content and Context?)

From Methodology 1.0 and with some new stuff, our scoring should have the following characteristics:
*   *Recognition* - whether an instance of data is recognized by our memory. 
    *   Recognition should be over a gradient - we should be able to say that something is recognized more or less than something else
    *   Recognition should be over generalizations - an unseen exemplar with a high-frequency prototype should be evaluated as recognized stronger than an exemplar we've seen twice
    *   A stronger way to do this is first filtering by basic level counts and then evaluating to see which is the most seen, and then filtering further by log probs
        *   **We should take all chunks whose basic level count is above threshold, and then find the best one by evaluating log-prob!!**
*   *Stability* - whether a chunk is built of two chunks that have been recognized prior over various occasions
    *   Not sure how to implement this yet, immediately my head goes to a separate threshold (higher-level) - almost like tiers! So tier 1 allows the chunk to be built, and tier 2 allows the chunk to contribtue to higher-level chunks (or recognition of candidate chunks vs. recognition of chunk stability)

Things we don't need:
*   *Fit* - I thought we needed this, but we mix context-information into the "recognition" categorize
*   *Reusability* - I thought we needed this, but again, recognition guarantees that by the time we create a chunk, we've seen it enough times to evaluate that we'll *probably* reuse it in the future

Generally, the two scores that we've cooked up are probably more than enough, but neither of them use the context. Still, it might be ok because context is stored inherently in the makeup of our content instances

### Order effects

Building the wrong chunks first probably has problems - we should find some way to learn the best order in which chunks should be built

This could also be a nod to the need for inside-outside parsing - we might need to monitor the capacity to build new chunks with each chunk we create (in the evaluation process)
*   This doesn't exactly feel intuitive to me because it makes sense that we build chunks via a frequency-rule, and assuming we keep that frequency rule, we shouldn't ever need to evaluate a chunk's prospect UNLESS we start to think about optimality as more than just frequency (this would warrant inside-outside parsing)

### Gathering distributional context

*   THERE ARE SO MANY WAYS TO DO THIS - and honestly, I think a distributional buffer for Cobweb is a generally valuable solution and should be discussed
    *   Instances are made up of an average of observations - this makes sense relative to how humans track instances as well
    *   We store distributional data relative to multiple things that we know are the same - ?!?!? We should have a preliminary short-term Cobweb hierarchy that feeds into a longer-term Cobweb hierarchy

### Refitting based on concepts

*   This is pretty self-explanatory - we need to represent nodes in our parse tree in terms of context relative to what it represents
*   An important note here is that order matters if we do this!! We should build concepts in terms of other concepts and reflect 

### A different data-structure??

Cobweb is SO GOOD for this purpose because it does literally all the things we need it to do - but hypothetically speaking, what is a list of the things we need to do? If, for instance, we don't end up using Cobweb...

Our data-structure must do the following:
*   Must be incremental on some level - or, at the very least, should be online
*   Must take in a discrete instance of some sort (I really like attribute-value representations)
*   Must yield a score of "recognition" which takes into account both frequency and accuracy
    *   As mentioned by the scoring data above, recognition should fulfill all those criteria (though the actual mechanism might be different)
*   Must contain a basic level of some form (which adapts as the data store expands) (or at the very least, some generalization)

### Removing Primitives?

One thing I've been thinking about thoroughly is putting primitives into a separate hierarchy, so that their representations don't affect the creation of chunks - however, there would once again arise a problem with nouns being paired with their relative clause more often than they were paired with the adjective

Grammars may have something to do with this - honestly, I don't really care whether we find chunks to be strong or weak as long as we build chunks consistently (recognition score is key here though)

## Methodology 1.1

*   Short Revision: what if categorization through a greedy, discretely chosen path is not enough to guarantee coherence in the long-term?? 
    *   Just see what we can extrapolate from multi-node contexts!!

Also added a ton of features like empty weighting, different weighting of surrounding nodes, etc

## Methodology 1.0

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
    *   We also replace the content hierarchy paths with actual context hierarchy paths (updated for the specific node) and then replace content-refs wherever applicable as well!!

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

    *   The process of generation should be as follows: sample a complex context instance, find the leaf which corresponds to its content-ref of the context instance, find the basic level node of that leaf, sample a new leaf from that node, expand its two content elements as new nodes by using PATH INFORMATION to traverse the CONTEXT HIERARCHY, and repeat this process until words terminate as sentences!!!
    *   For generation with masked language, we do the same generation process above for each masked token (but we use surrounding context to find the initial context-hierarchy node as well)
        *   One small clarification to this - what we should do is we should build a new composite node next to the child node, and then use the context from previous parse to predict what that new composite node's content is, and then decompose it in the same way we would decompose a new generation from scratch

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
    *   complexity = 1 for all Primitive Nodes
    *   index - index of the current content (for calculating and visualization)
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
    *   index - index of the current content (for calculating and visualization)
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
    *   Visualize methods for visualizing the parse tree
        *   I would like for nodes to be represented by circles, and clicking on them expands both the content-instance and the context-instance at the same time
    *   to_json(filename) and from_json(filename) methods

Rolling Parse Tree: TBD

Long Term Memory:
*   Variables
    *   context_hierarchy = CobwebDiscreteTree
    *   content_hierarchy = CobwebDiscreteTree
    *   corpus
    *   id-to-value
    *   value-to-id
*   Methods
    *   Helper methods:
        *   instance to id-instance
        *   id-instance to instance
    *   get_content_instance_statistics()
        *   A bit like the current FiniteParseTree.score_function - gives all the statistics we'll need to do thresholding and score calculation
    *   get_context_instance_statistics()
        *   Same as above, but for the context hierarchy
    *   update_vocabulary(actions)
        *   Given the actions from a Cobweb ifit, make changes to vocabulary and the Cobweb hierarchy accordingly

Webster: (Primary Class - we're going to do all logic and parsing in here, and every other class is simply going to be a data-class that stores necessary information and helper methods)
*   Variables
    *   Corpus, context size, threshold
    *   An instance of the Long Term Memory
*   Methods
    *   parse_sentence(sentence, threshold=None, new_vocab=True, learning=False)
        *   Creates a parse tree for the sentence.
            *   We are in charge of maintaining the frontier of candidate chunks here
        *   threshold determines the threshold at which chunks are added
        *   new_vocab determines if new vocab is in the sentence
        *   if learning == true, we add the sentence at the end
            *   We should already have the top level of candidate chunks
        *   This method will manage all of the logic
    *   evaluate_chunk(content_instance, context_instance) - a method that, given the content and context instances for a candidate chunk, returns the necessary scores for that thing
        *   This should just call methods from the long-term hierarchy and do calculations / thresholding for heuristics
    *   generate_sentence(masked_sentence)
        *   masked_sentence is of the form "word1 word2 [mask] word3..." and predicts an expansion for each masked token available
        *   Masked tokens are not part of the vocabulary, but instead an indicator to do sampling from basic-level nodes

## Things to take note of

*   Need to figure out a context-only weighting style that works (perhaps negative attributes, sorting only by context should yield some valuable results)
    *   Perhaps bag-of-words-style weighting resurfaces here! Honestly, probably not - adjective order ruins the placeholders here which sucks, and a sliding window feels more feasible especially in accordance with the other stuff
    *   VERBALIZED: the reason that bag-of-words is bad is because it's not enough to explain the discrete relation that we set in stone here

*   Remember that this requires complexity for the generation and chunk building processes, cool things that we can do with stability here!

*   It might be true that building chunks requires both an understanding of context and content - HOWEVER, if we define every content item purely by its context (which I think is super reasonable) every chunk combination is naturally a context-driven creation!

*   Still a problem with generation because our pre-fit labels are not robust - this needs fixing before we commit to only using pre-fit labels. I have post-fit labels as a fix to this, but we'll need to address

# **Bonus:** Parallels to Diffusion Models, BPE + Tokenization

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
        *   If this system ever succeeds at a single-hierarchy implementation, we can totally do this
        *   Generally, the context hierarchy can probably be used to handle this as well 
    *   Another valuable extension here could be if we do a masked representation and then 

*   An important note comes into play when discussing this is whether semantic information will be appropriately translated, and I think that diffusion models will fare better than our purely discrete counterpart in using the latent space to leverage both semantic and syntactic correlation (contextually related)

*   Another note here is that while our current process resembles diffusion LMs, the continuous parse tree is more analogous to autoregressive models!!

MOVING THIS TO ITS OWN REPOSITORY SO THAT WE CAN START WORKING ON IT!!!

## Rationale for this theory

*   There's not really a problem, it's just that context gets treated weirdly and I think that separating into two hierarchies will make analysis a little cleaner and strengthen our chunk creation
*   It's also going to be a LOT easier to program a heuristic that makes sense and is adaptable
*   We've also evaluated that tracking stability and complexity of instances is incredibly important, and I think that the two hierarchy-partition will encode metadata appropriately

*   *Bonus but fun!* - splitting into two hierarchies will almost guaranteed make the diffusion model analogy a lot easier.