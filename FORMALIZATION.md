# Formalizations of the Chunking Framework

## Intuition

Pat can probably fill this part in better than me, but the idea is almost a "compression" of memory by observing chunks and their compositionality. More frequent chunks should be compressed more often than less frequent chunks!

**NOTE:** CobwebTree(1e-4, True, 0, True, True) is a good parameterization!

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

Note that this implementation can be done with one hierarchy where nodes are tagged according to their real-ness - we really need to figure out three main questions:
*   When a chunk goes from being virtual to being real
    *   We need to organize some form of frequency cutoff based on Cobweb with respect to the amount of instances
    *   This change needs to be performed by a path aggregation of SOME SORT
    *   Although this feels like a CLASSIC basic-level definition application, we can get around this by finding the mean value of the sums of the counts of the nodes along the path that we categorize to in the virtual hierarchy (**I think we're onto something here!**)
*   Which chunk is the best real chunk, of the real chunks categorized (note that this is only relevant if our prospective chunk definitions are overlapping)
    *   This can be done quite simply according to our current metric, which scores the given chunk with respect to what it's seen
    *   I'm not actually sure what the "best" chunk is and am hoping we avoid this altogether, but we know that a chunk is a good fit if we've seen it before
    *   This does raise a bit of a larger concern - perhaps we need to score based on the pathsum
*   We need to define whether chunks are real or virtual to begin with
    *   The threshold for when a virtual chunk becomes real is probably a good step here - we can find out whether that's good enough to establish value
*   We need to figure out the best way to term the idea of "recognition" - we can use this to define a lot of the above works
    *   A normalized log-probability will do the trick quite well I feel (according to some preliminary conversations with Pat) but we really need to lock in on that

We're also going to program the change where we separate context into individual attributes in addition to content for positional encoding and further separation for clarity. This should separate our chunks cleanly.

**NEW and TESTABLE generalization:**
*   Can we just add every possible candidate to our long-term memory? This would consist of all valid parses plus all top-level candidates
*   We're probably going to need to adjust the scoring mechanic on some level to include a frequency-based aggregation by the path sum, and we'll also need a separate score to confirm that our categorization is in fact good
    *   Our separate score already exists in the form of an averaged log-probability
    *   Let's start by adding all of the potential candidate chunks at the root level to a long-term hierarchy and then figuring out what heuristics we can derive from there
    *   We basically just start programming this heuristic and then hopefully we can extend it to long-term data

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

## Hypotheses and results

