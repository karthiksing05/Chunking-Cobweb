# Formalizations of the Chunking Framework

## Intuition

Pat can probably fill this part in better than me, but the idea is almost a "compression" of memory by observing chunks and their compositionality.

## Processes

There are two main processes that go into our framework: a process for generating parse trees and a process for adding them to our long-term memory. We will describe the needs for each of these and then describe how we validated them.

First, we believe the parse tree generation process should satisfy the following criterion:
*   Previously seen windows of content should be parsed in the same structure if seen again (deterministic parsing)
*   Parse trees structure should rely on interpolations of commonalities in previously seen parse trees

We will also outline the following criteria for the adding to long-term memory:
*   Parse trees should be added to the long-term memory such that the long-term memory groups similar chunks together.

In order to generate parse trees in the style of prior parse trees, we need to create a scoring metric that matches chunks that have been seen before based on content, context, or both. The most likely chunk will always be added first.

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

## Implications

*   PRODUCT QUANTIZATION WITH COBWEB! This is a real thing we can bring forward as a method of either generating or finetuning embeddings