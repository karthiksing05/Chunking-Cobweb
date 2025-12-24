# Formalizations of the Chunking Framework

## Intuition

Pat can probably fill this part in better than me, but the idea is almost a "compression" of memory by observing chunks and their compositionality. More frequent chunks should be compressed more often than less frequent chunks!

**NOTE:** CobwebTree(0.1, False, 0, True, False) is a good parameterization!

## Processes

There are two main processes that go into our framework: a process for generating parse trees and a process for adding them to our long-term memory. We will describe the needs for each of these and then describe how we validated them.

First, we believe the parse tree generation process should satisfy the following criterion:
*   Previously seen windows of content should be parsed in the same structure if seen again (deterministic parsing)
*   Parse trees structure should rely on interpolations of commonalities in previously seen parse trees

We will also outline the following criteria for the adding to long-term memory:
*   Parse trees should be added to the long-term memory such that the long-term memory groups similar chunks together.

In order to generate parse trees in the style of prior parse trees, we need to create a scoring metric that matches chunks that have been seen before based on content, context, or both. The most likely chunk will always be added first.

### New addition: Virtual and Real Chunks!

From discussions with Pat on 11/21/2025, we have made some very important distinctions! I'm going to list the rationale first and then the specific implementation for our rationale.

**Rationale:** our framework at the given state it was in could parse chunks based on existing chunk structures it had seen before. However, our framework could not generalize new chunks or find a way to jump-start the chunking process from scratch, a vital need. After briefly delving into approaches of other algorithms, I found a lot of trivial rules for establishing "chunks" - most were based on trivial frequency-based cutoffs and varied with calculated statistics. Furthermore, we needed to establish higher-level chunks, which are made up of chunks labeled not just by a single discrete label, but a list of labels representing the path taken to arrive at that instance being sorted. Therefore, a simple table may not suffice cleanly. However, we still do need to aggregate statistics to represent 

**Solution:** we arrived at the conclusion that in addition to tracking statistics for established (*real*) chunks, we also need to track statistics for proposed (*virtual*) chunks. Our idea is as follows:
*   We start by initializing two Cobweb Hierarchies
    *   One that represents a portion of our long-term memory which categorizes and retrieves *real chunks*, which we'll denote as the *real hierarchy*
    *   One that represents a portion of our long-term memory which hierarchizes the *virtual chunks*, which we'll denote as the *virtual hierarchy*
*   For a given iteration, we parse through all available *candidate chunks*:
    *   Initialize two empty lists: a list of *valid candidates* and *virtual additions*
    *   For each *candidate chunk*:
        *   We first determine whether the chunk is *real* or *virtual*
        *   If we evaluate that the candidate chunk is *real*:
            *   We can save this chunk to our list of *valid candidates*
        *   Otherwise, we must conclude that the candidate chunk is *virtual*:
            *   We add this candidate chunk to the list of *virtual additions*
    *   If we have a nonzero amount of *valid candidates*:
        *   We add the best valid candidate to our parse tree and continue the iterations
    *   If there are no *valid candidates*:
        *   We break from the loop, saving the *virtual additions* globally
*   At the end, we do the following:
    *   For each chunk that is an *virtual addition*:
        *   We add it to the *virtual hierarchy*
        *   We collect statistics on the addition and determine whether that chunk is good-enough to be categorized a *real chunk* - if so, we add it to the *real hierarchy*

At our core, we're trying to only add chunks once we've aggregated enough frequency on them. It goes back to our original question on the necessity of chunks to begin with - chunks are, among other things, a way of compressing information in a valuable way. It follows intuitively that you would want to compress information such that the most frequently seen pattern are compressed often.

We really need to figure out three main questions:
*   When a chunk goes from being virtual to being real
    *   If a chunk is recognized in the virtual hierarchy, we can say that it's real?
*   Which chunk is the best real chunk, of the real chunks categorized (note that this is only relevant if our prospective chunk definitions are overlapping)
    *   This can be done quite simply according to our current metric, which scores the given chunk with respect to what it's seen
    *   I'm not actually sure what the "best" chunk is and am hoping we avoid this altogether, but we know that a chunk is a good fit if we've seen it before
    *   This does raise a bit of a larger concern - perhaps we need to score based on the pathsum
*   We need to define whether chunks are real or virtual to begin with
    *   We can do this through a recognition in the real hierarchy - assume that all chunks which are not seen in the real hierarchy are virtual
*   We need to figure out the best way to term the idea of "recognition" - we can use this to define a lot of the above works
    *   From discussions with Chris + Pat - the score function now is the maxed log-likelihood along all of the nodes along the path

#### **TESTABLE GENERALIZATION: SUCCESS!!**
*   Rather than creating two separate hierarchies, we create a single hierarchy and not only add all chunks to that hierarchy, but all unparsed "virtual" candidate chunks as well - we then let our recognition metric derive from there
*   I'm lowk a big fan of this as it gets around the multiple hierarchy approach and requires MINIMAL code changes to our existing framework.
*   One thing that we need to do which is relatively important is find what thresholds work as well as a formula (theoretical or empirical) for thresholding but this looks GREAT
*   In retrospect, this actually makes perfect sense, because we are storing the probability of a given chunk existing using Cobweb as the hierarchy, and by adding all unparsed instances, we are letting probabilities for each of them and their generalizations accumulate over time.
*   **NOTE**: our recognition mechanism may be flawed!
    *   Recognition is defined (kind of) but is honestly only in terms of direct utility due to the root node being an estimator for other nodes. I'm not really sure how this scales with respect to more complex nodes for which there's only a partial match
    *   We may need to reframe the recognition metric in terms of classifying where the "homes" of each node is - the node with the most plausible "home"
    *   "Homes" should be ranked by what families are seen more often - the tradeoff is obviously between finding a "home" that is descriptive of its instances as well as a 

#### **Actually programming the plan: "Chunk Utility"?!?**
*   To recap, we are trying to decide which chunks to "realize" of the virtual chunks.
*   Unfortunately, the testable generalization was only one piece of the puzzle. In order to truly program a chunking architecture, we need a formula to recognize the best chunk at each iteration that goes beyond just a simple frequency measure.
    *   I propose here that the *best chunk* to realize at any given time is found by maximizing two qualities: **frequency of proximity** and **reduction of overall complexity**.
    *   **Frequency of proximity** of course refers to how often some things have been seen, close together in the same way. My logic is that content elements that are seen in association more frequently should be chunked rather than content elements rarely seen together, as one donates value that the other does not.
    *   **Reduction of overall complexity** is a newly added criteria from the testable generalization added above - when presented with the choice to realize two chunks which have been seen relatively often, we want to make the decision to realize the chunk that has the highest possibility of reuse. In other words, 
*   Chunks are a theory of information compression in some sense. Both of the above criteria must be realized in order to properly satisfy information compression in a way that is intentional and yields simple chunk grammars. Right now, our testable generalization creates grammars that are more complex than necessary, which may result in coherent output, but also may not hold under semantic additions.
---
*   Our goal should be to add a heuristic that mitigates the reduction of overall complexity - i.e. in my opinion, we should add a new chunk **only when we have to**.
*   Thus, I propose a new metric called **chunk utility**, the combination of a score of recognition and resistance towards creation.
*   How can we define resistance towards creation? (As a side note, this will be considered with two hierarchies in mind)
    *   This may requit sorting down the tree similar to the `ifit` method with learning turned OFF, where we evaluate very strictly the possibility of progressing down the tree and terminating or selecting a new node
    *   Perhaps, instead of categorizing, we do a BEST/NEW fit and check the counts of the nodes that we add to?
    *   We create all possible chunks in the virtual hierarchy but moving them takes resistance
        *   Resistance is gradually lowered by the recognition threshold! That way, if no chunk is recognized, we slowly drop our barrier!!!
*   What if we just move prototypes of good chunks to the real hierarchy - when adding chunk instances, if the instance is recognized in the real hierarchy then we're good otherwise we add to virtual hierarchy?
    *   Perhaps the datastructure for storing real chunks is not a hierarchy?? Must consider all possibilities
    *   For this to work, recognition must be binary! We must either recognize or not recognize stuff, and that may requit 
    *   After we add a new item to the virtual hierarchy, we can check to see if anything in its path (except the root node) which has not already been moved needs to be moved?
*   One other thing - do we ever need to project future instances? This may be relevant but it also may not be (for recursive structure) as any stage of recursive expansion can be noticed

### **New plan for advanced grammars, recursion, everything!**
*   Recursion solution:
    *   We are never sorting the base case!!! For the sentence "The ___ runs", we can use "big bat, big red bat, big red furry bat" but we never know to use "bat"
    *   The solution here is definitely to add primitives to the same hierarchy as everything else and hope that context sorts it out
        so that we know what's at the bottom!!! This is beautiful!
    *   We also definitely have to sort everything in the same hierarchy so that we can store common senses!
    *   [OPTIONAL] we may need to go back to using a bag of words here so that we can scrape meaning!
*   Plan for primitives:
    *   Sort primitives at the same time as candidate chunks? Or create a stable hierarchy of primitives and then add candidate chunks to it?
    *   I think we should sort primitives until we have some sort of convergence there (recognition of primitives, as they are chunks too!) - let them go through the same process of recognition and chunk-building!!!
        *   Once we recognize chunks, we can move to the next layers of chunking and we can simply just extend the philosophy for primitive instances
        *   Do we use path-annotated context or just regular context? I think because of the way we've been handling composite chunks we should just do regular context and give content elements the path information
*   Chunk utility solution:
    *   We are normalizing the average log-probability of the whole path, but with an "information-style" weighting. Each log-probability gets halved as you move down the path so it's log-probability at the root with some extra steps.
    *   My hope is that we can threshold by some measure of uncertainty that is particular to the path that we categorize down itself
    *   There may be something interesting we can do with entropy in order to properly yield a "stability" focused calculation - higher instances should have more entropy and lower instances should have less entropy
        *   It's essentially variance so I'm not sure if it by itself can measure something, but perhaps a count-scaled entropy could be valuable
    *   I completely disagree with Chris - I think that path information will yield very important semantic relevance and it's similar to how we as people store information. At any given point, we need to be able to recall that a robin is both a bird and an animal so that we can make efficient decisions based on how it appears within the sentence\

#### *Implementation Plan*
*   Program primitives first!!
*   Normalizing the score needs a lot of work - we need a binary metric of recognition

## Design Decisions

One of the most important things to keep track of over the course of this project is the importance of labeling and defining design decisions. These decisions are influenced and ascertained by any choice / assumption made over the course of defining the framework. While current design decisions have been made to support ease of access regarding the programming of the framework, they'll definitely be revisited as we aggregate results and rationale from our tests.

*   The distinction between primitive and composite instances is a pretty unique design decision - additionally, the decision to include more than one concept element and the relations we've fixed (currently sequential, left-right) can be changed as well
*   How is the most optimal chunk candidate added?
    *   I think this is currently being done by log-probability - the chunk candidate that best matches a concept within the long-term hierarchy is added to the tree (each candidate is categorized to identify the best possible match)
    *   In future instances, we could do this by some kind of collocation score?
*   How do we select the best chunk candidate label?
    *   Currently done by basic-level nodes, evaluated on the log_prob_instance_missing metric! There are talks of a Pointwise Mutual Information metric coming into play but subject to change.
    *   THIS HAS CHANGED - we're now just passing in the whole path and letting Cobweb ascertain which node gets the best path!
*   How do we denote when parse tree construction terminates?
    *   Partial parse tree based on the metric by which we add nodes to the parse tree
    *   Full parse tree every time until one node reached
*   The decision to include the SPLIT / MERGE operations is incredibly interesting
    *   Prior studies done with Cobweb/4L indicate that SPLIT and MERGE were barely used but that they definitely could be in a longer term
    *   SPLIT and MERGE contribute to long-term inference efficiency (removing of redundant concepts) which is super relevant for beginning and end behavior
*   The data we choose to read in is subject to change - currently, we're reading language by sentences, but future iterations can do larger-scale windows and slide the window / create parse trees flexibly over time.
    *   We could read in the first n primitives and parse them, then we read the next n primitives (keeping the root node of the primitives)
    *   [COOL STUFF] If we construct partial parse trees, we can continue iteratively adding primitives to our "active working memory" until the threshold for working memory has been met
        *   We can either dump everything at once OR add nodes to LTM as we build them. I am inclined to believe in the latter.

## Replicating LLM Technology

We're going to need to design a gauntlet that truly lets this learn language - my hope is that a batch-learning case fails but an incrementally additive grammar allows this to improve to the skill of an LLM. (i.e. learning nouns and verbs and articles first, then adjectives, then more complex sentences, then maybe different tenses, etc. etc.)
