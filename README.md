# Chunking Cobweb

Work with Dr. Pat Langley and Dr. Chris Maclellan on ISLE Internship to model attributes of the psychological principle of cognitive chunking through a Cobweb-backed framework.

The formal writeup can be found in `FORMALIZATION.md`, under which I describe the design decisions, hypotheses, and evaluations that result in bringing forth this cognitive architecture as a solution to new 

## Requirements + Setup

All requirements can be found under requirements.txt, save for cobweb, which must be installed through the Git repository so that we can make necessary changes to it.

Run the following commands:
```sh
# install all necessary prerequisites
pip install -r requirements.txt

# install and modify cobweb
git clone https://github.com/Teachable-AI-Lab/cobweb
cp my_cobweb_discrete.cpp cobweb/src/cobweb_discrete.cpp
cd cobweb
pip install .
cd ..
```

After running these commands, you should be fully set up with the modified install of Cobweb!

## Guiding Intuition

[TODO] DO A WRITEUP HERE OF HOW THE FRAMEWORK WORKS!
[TODO] NEED TO TALK ABOUT SCORING HERE AS WELL!
*   

## GUI Editor:

Because of the highly conceptual nature of this framework, we've also created a GUI by which we can inspect and analyze both parse trees and our long-term memory.

*   ```gui/parse_tree_editor.py``` - a stylistically consistent editable parse tree generator that can create and export parse trees.
    *   Through this editor, we can show the learning mechanism to be effective through a variety of tests that confirm consistency and stability through fed parse trees.
*   ```gui/ltm_inspector.py``` - an also stylistically consistent long-term-memory (CobwebTree) inspection tool that can peruse and analyze the long-term memory after a given length.
    *   [TODO] this is still in progress! but subject to change whether we even need this ngl (need a better LTM representation for sure)

## Important Tests:

Below is a list of tests confirming specific value propositions behind the chunking framework. They are designed to test and demonstrate the capability of "stably defined" long-term memories

*   ```demo/parse_consistency_test.py```
    *   Demonstrates that parses have consistency with subparses within the tree
    *   Showcases the robustness of the framework against catastrophic forgetting over long-term parses
*   ```demo/parse_consistency_scale_test.py```
    *   Trying to answer the question: at what point is our long-term memory "stable"?
*   ```demo/time_complexity_tests```
    *   ```demo/time_complexity_tests/document_scaling_test.py```
        *   [TODO]
    *   ```demo/time_complexity_tests/parsing_scale_test.py```
        *   [TODO]
    *   ```demo/time_complexity_tests/grammar_scaling_test.py```
        *   [TODO]
*   ```demo/vocab_expand_test.py```
    *   [TODO]
*   ```demo/grammar_catching_test.py```
    *   [TODO]
*   ```demo/ltm_analysis.py```
    *   A test that analyzes various long-term memories as well as some intermediate subtree levels.
    *   We hope to see a resulting taxonomy that alternates between AND-like nodes defined mostly by content and OR-like nodes defined mostly by context.
*   [TODO] Need more tests in general!

## Unit Tests:

Below is a list of all tests confirming the syntactic consistency of the framework. Use ```pytest -s tests/TEST_NAME_HERE.py``` to run (the "-s" flag is for output)

*   ```unittests/parse_tree_test.py``` - a test to confirm the correct implementation of parse trees and parse tree composition
*   ```unittests/gen_learn_test.py``` - a test to also confirm the logic of parse tree addition and processing the parse trees into the long-term memory
*   ```unittests/partial_parse_analysis.py``` - a test used to inspect different thresholds at which to cut 

## Updates to Cobweb:

A couple modifications were made to Cobweb to support some of the edge-case behavior of the chunking framework. We leave these changes in ```my_cobweb_discrete.cpp``` - when running this code, copy the contents of that file into the ```cobweb/src/cobweb_discrete.cpp```.

Full list of changes is below:

*   Mode 4 - BEST + NEW
    *   Created a new mode, signified by the mode=4 argument passed to CobwebTree.ifit, that only evaluates either the BEST or NEW actions in the event that no edge cases are evaluated. In other words, it completely ignores the MERGE and SPLIT actions, and this is done to make sure no nodes within the Cobweb hierarchy are deleted and speed up processing time - MERGE and SPLIT actions take up less than 10% of all actions across most Cobweb trees.
    *   The hierarchy is capable of running without this setting now, but it was initially created to test other features.
*   Made AV_Counts editable
    *   For each node, I want to modify the AV-count variable. Under the current C++ implementation, it's locked as "read-only".
    *   I created a method, set_av_count, which overwrites a CobwebNode c's c.av_count attribute with a passed-in Python dictionary.
    *   This method has the potential to break the tree completely, but it is applied recursively from leaf-to-root for any given node path, so probabilities are not destroyed in any given path.
*   Made Concept_IDs editable
    *   Similar to AV_Counts, having access to the concept hash of a given node will help us with memory storage and then 
*   Modify Cobweb such that it contains a DEBUG mode (as parameterized by a "debug" argument) and logs all create/delete actions (by concept hash, new and old)
    *   This is a purely stylistic / debugging change, and I'll provide a code snippet below that details how concepts should be logged and analyzed. This change should primarily happen in the "ifit" method with the logging categorizing changes before and after the enactment. 
    *   Each entry is a dict, e.g.:
        *   {"action": "NEW", "node": "abc123", "parent": "def456"}
        *   {"action": "MERGE", "new_node": "ghi789", "parent": "root000", "children": ["abc123", "xyz999"]}
        *   {"action": "SPLIT", "deleted": "abc123", "parent": "root000", "promoted_children": ["child1", "child2"]}
        *   {"action": "BEST", "node": "ghi789"}
    *   All edge cases caught before the main evaluation should be classified as "NEW" as well!
*   Update to categorization logic - path-focused probability matching
    *   Rather than matching discretely by category, we need to make sure we have some distance-based metric of judging whether two categories are "close" or "far". Thus, we'll be using the paths of each concept label to match how close a given node is to a given other node!
    *   To do this, we'll be calculating a "flattened" frequency count with the weighted path of the node we categorize. The process is as follows:
        *   Chunk candidates will be labeled by the leaf node they are categorized under, but the content value they donate to a chunk instance that relationally merges them will be a dictionary of the entire path with frequencies weighted by distance along the path (relative to where in the path you are, similar to how we did PathSum for Cobweb)
            *   We'll mess around with different path weightings - another valid hypothesis is that the uniform weighting may be successful
        *   Now we are able to compute content similarities optimally by checking the similarities and overlaps along a path of frequency!
    *   We're doing this because in the potential case where frequencies (and as such, probabilities) exist for more than one concept-id leaf, we can categorize according to the merged frequencies across the nodes (I'm assuming that the implementation calculates probabilities on the fly based on frequencies for actual comparison - hopefully the comparisons don't change the value)
    *   The additional hope is that this preserves basic-level definitions by just assuming all overlapping levels contribute to the matching process!

## Long-term Goals

After we effectively demonstrate that this system can cleanly parse language and related associations, we can implement this system in more puzzle frameworks. Word searches, Chess, and Gomoku (which TAIL already implements) are all valuable implementations.

Additionally (and this is more tentative) creating an embeddings prior that overlaps the tree structure can potentially result in more comprehensive embeddings that are inherently based on relation and composition rather than relying on association and emergence.