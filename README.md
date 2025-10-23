# Chunking Cobweb

Work with Dr. Pat Langley and Dr. Chris Maclellan on ISLE Internship to model attributes of the psychological principle of cognitive chunking through a Cobweb-backed framework.

## Requirements + Setup

All requirements can be found under requirements.txt, save for cobweb, which must be installed through the Git repository so that we can make necessary changes to it.

Run the following commands:
```
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
*   

## GUI Editor:

Because of the highly conceptual nature of this framework, we've also created a GUI by which we can inspect and analyze both parse trees and our long-term memory.

*   ```gui/parse_tree_editor.py``` - a stylistically consistent editable parse tree generator that can create and export parse trees.
    *   Through this editor, we can show the learning mechanism to be effective through a variety of tests that confirm consistency and stability through fed parse trees.
*   ```gui/ltm_inspector.py``` - an also stylistically consistent long-term-memory (CobwebTree) inspection tool that can peruse and analyze the long-term memory after a given length.
    *   [TODO] this is still in progress! but subject to change

## Important Tests:

Below is a list of tests confirming specific value propositions behind the chunking framework. They are designed to test and demon

*   ```demo/parse_consistency_test.py```
    *   Demonstrates that parses have consistency with subparses within the tree
    *   Showcases the robustness of the framework against catastrophic forgetting over long-term parses
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
*   [TODO] Need more tests in general!

## Preliminary Tests:

Below is a list of all tests confirming and acknowledging the effectiveness of the framework. Use ```pytest -s tests/TEST_NAME_HERE.py``` to run (the "-s" flag is for output)

*   ```tests/parse_tree_test.py``` - a test to confirm the correct implementation of parse trees and parse tree composition
*   ```tests/gen_learn_test.py``` - a test to also confirm the logic of parse tree addition and processing
*   ```tests/ltm_analysis.py``` - a test that analyzes various long-term memories as well as some intermediate subtree levels.

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
*   Update to categorization logic - path-focused
    *

## Design Decisions:

One of the most important things to keep track of over the course of this project is the importance of labeling and defining design decisions. These decisions are influenced and ascertained by any choice / assumption made over the course of defining the framework. While current design decisions have been made to support ease of access regarding the programming of the framework, they'll definitely be revisited as we aggregate results and rationale from our tests.

*   The distinction between primitive and composite instances is a pretty unique design decision - additionally, the decision to include more than one concept element and the relations we've fixed (currently sequential, left-right) can be changed as well
*   How is the most optimal chunk candidate added?
    *   I think this is currently being done by log-probability - the chunk candidate that best matches a concept within the long-term hierarchy is added to the tree (each candidate is categorized to identify the best possible match)
    *   In future instances, we could do this by some kind of collocation score?
*   How do we select the best chunk candidate label?
    *   Currently done by basic-level nodes, evaluated on the log_prob_instance_missing metric! There are talks of a Pointwise Mutual Information metric coming into play but subject to change.
*   How do we denote when parse tree construction terminates?
    *   Partial parse tree based on the metric by which we add nodes to the parse tree
    *   Full parse tree every time until one node reached
*   The decision to include the SPLIT / MERGE operations is incredibly interesting
    *   Prior studies done with Cobweb/4L indicate that SPLIT and MERGE were barely used but that they definitely could be in a longer term
    *   SPLIT and MERGE contribute to long-term inference efficiency (removing of redundant concepts) which is super relevant for beginning and end behavior
*   The data we choose to read in is subject to change - currently, we're reading language by sentences, but future iterations can do larger-scale windows and slide the window / create parse trees flexibly over time.
    *   We read in the first n primitives and parse them, then we read the next n primitives (keeping the root node of the primitives)
    *   [COOL STUFF] If we construct partial parse trees, we can continue iteratively adding primitives to our "active working memory" until the threshold for working memory has been met, and then we can dump everything into the memory all at once?

## Long-term Goals

After we effectively demonstrate that this system can cleanly parse language and related associations, we can implement this system in more puzzle frameworks. Word searches, Chess, and Gomoku (which TAIL already implements) are all valuable implementations.

Additionally (and this is more tentative) creating an embeddings prior that overlaps the tree structure can potentially result in more comprehensive embeddings that are inherently based on relation and composition rather than relying on association and emergence.