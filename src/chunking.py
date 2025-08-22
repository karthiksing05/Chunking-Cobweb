"""
The primary class of the main datastructure, LanguageChunkingParser.
"""

from cobweb.cobweb_discrete import CobwebTree, CobwebNode
from src.viz import HTMLCobwebDrawer
from src.parse import ParseTree
import re


class LanguageChunkingParser:

    """
    This is only for language right now - lots of changes will need to be made
    to generalize, such as properly representing context.

    Standardizations:
    *   id-0 => content-left
    *   id-1 => content-right
    *   id-2 => context-before
    *   id-3 => context-after
    """

    def __init__(self, value_corpus, context_length=3):

        self.ltm_hierarchy = CobwebTree()
        self.cobweb_drawer = HTMLCobwebDrawer(
            ["Content-Left", "Content-Right", "Context-Before", "Context-After"],
            value_corpus
        )

        self.id_to_value = [x for x in value_corpus]
        self.value_to_id = dict([(w, i) for i, w in enumerate(value_corpus)])
        self.id_count = len(value_corpus)

        # adding root node to dictionary! edge case not properly counted
        hash = self.ltm_hierarchy.root.concept_hash()
        self.value_to_id[f"CONCEPT-{hash}"] = self.id_count
        self.id_to_value.append(f"CONCEPT-{hash}")
        self.id_count += 1

        self.context_length = context_length

    def parse_input(self, sentences, end_behavior="converge", debug=False):
        """
        Primary method for parsing input (a list of sentences) and updating the
        long-term-memory hierarchy using a parse tree.

        Returns the Parse Tree!
        """

        parse_trees = []

        for sentence in sentences:
            parse_tree = ParseTree(self.ltm_hierarchy, self.id_to_value, self.value_to_id)
            parse_tree.build(sentence, end_behavior, debug)

            parse_trees.append(parse_tree)

        return parse_trees

    def add_parse_tree(self, parse_tree):
        """
        Method to add the parse tree to the long-term hierarchy.

        Important note here is that because the parse tree doesn't contain the
        primitive instances in code (the leaves of the code tree are composite
        by nature), we can literally just add all nodes of the parse tree as
        defined by the code.

        Pipeline:
        *   Save tree structure before adding all new nodes (for each node,
            what is its parent and what are all its children)
        *   Add all instances of the parse tree as children via Cobweb's ifit
        *   Now, we iterate over the new tree and the old tree and keep track of
            all fundamentally applied actions.
        *   We initialize an empty list, "rewrite_rules", for updating all
            instance dictionaries after properly logging all actions.
        *   We process each Cobweb action in an effort to keep an updated as
            follows:
            *   Any "Add" operations don't change the tree's node structure and
                are not necessary to process.
            *   Any "Create" operation creates a new node (typically the last
                action) - these nodes must be added via the pattern
                "CONCEPT-{node.concept_hash()}" to the self.value_to_id and
                self.id_to_value dictionaries (also update self.id_count).
            *   Any "Split" operation deletes a node and promotes all children
                to become parents - I'm PRETTY SURE there's no vocabulary
                changes, but we will need to save a rewrite rule binding the old
                node vocabulary id to it's parent's id, (deleted_id, parent_id).
            *   Any "Merge" operation groups two children and creates a common
                parent node for these two nodes. In this case, we should add the
                new node to the vocabulary similar to the "Create" action.
        *   Finally, we iterate through the tree and apply all rewrite rules
            (which I'm pretty sure are only SPLIT rules, so this should be
            fairly easy).

        Additional Notes:
        *   We're currently just comparing the old tree to the new tree to
            safely make all changes, but a future implementation will rewrite
            the Cobweb class to natively return these actions for our
            convenience and faster time-complexity.
        *   In the long term, we should keep track of "stale" concept ids in our
            vocabulary to replace them over time so our vocabulary doesn't
            balloon, but again, a later fix.
        *   Finally, rewrite rules may not currently exist for created nodes but
            there is a layer of specificity that we can and should replace.
            Still not sure about the best way to go about that, but leaving it
            here for future notice.
            *   In general, the presence of a rewrite rule makes the
        """

        """
        Important prior states to save:
        *   Each node's concept hash
        *   Each node's children
        *   Each node's parent
        """

        # saving old tree structure
        prior_parents = {}
        # prior_children = {}

        to_visit = [self.ltm_hierarchy.root]

        while len(to_visit) > 0:
            curr = to_visit.pop(0)

            if curr.parent:
                prior_parents[curr.concept_hash()] = curr.parent.concept_hash()
            else:
                prior_parents[curr.concept_hash()] = None

            # prior_children[curr.concept_hash()] = None
            # if curr.children and len(curr.children) > 0:
            #     for child in children:
            #         prior_children[curr.concept_hash()] = curr.parent.concept_hash()

        # adding all new instances
        insts = parse_tree.get_chunk_instances()

        for inst in insts:
            self.long_term_hierarchy.ifit(inst)

        # saving new tree structure
        curr_parents = {}

        to_visit = [self.ltm_hierarchy.root]

        while len(to_visit) > 0:
            curr = to_visit.pop(0)

            if curr.parent:
                curr_parents[curr.concept_hash()] = curr.parent.concept_hash()
            else:
                curr_parents[curr.concept_hash()] = None

        """
        Keeping track of all interactions with logic - hopefully with Cobweb
        supporting a debug mode, this is made better!

        We just need to keep track of all newly created and deleted nodes and
        adjust them according to our logic above!

        For created nodes:
        *   If the node is created:
            *   Add it to the vocabulary!
        For deleted nodes:
        *   If the node is deleted:
            *   Iterate through all nodes and replace the node's id with its
                parent's id.
        """

        # created nodes
        for hash, parent in curr_parents.items():
            if not hash in prior_parents:
                new_vocab = f"CONCEPT-{hash}"
                self.value_to_id[new_vocab] = self.id_count
                self.id_to_value.append(new_vocab)
                self.id_count += 1

        # deleted nodes
        rewrite_rules = []

        for hash, parent_hash in prior_parents.items():
            if not hash in curr_parents:
                rewrite_rules.append((f"CONCEPT-{hash}", f"CONCEPT-{parent_hash}"))

        # replacing rewrite rules

        to_visit = [self.ltm_hierarchy.root]

        while len(to_visit) > 0:
            curr = to_visit.pop(0)

            inst_dict = curr.av_count

            for _, v in inst_dict.items():
                for key, _ in v.items():
                    for old_key, new_key in rewrite_rules:
                        if old_key in inst_dict:
                            inst_dict[new_key] = inst_dict[old_key]
                            del inst_dict[old_key]

            curr.av_count = inst_dict

        return True

    def visualize(self, out_base="cobweb_tree", display_in_colab=True):
        """
        We had a rudimentary CobwebDrawer before but I'd very much enjoy if we
        could expand on this and create an HTML-drawing Cobweb method before we
        continue tests - it would be both easier to explain and certainly easy
        to verify.
        """
        pass