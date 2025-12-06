"""
Vocabulary Generation!

The premise for this comes from categorizing, for example, chunks that operate over the following
set of instances:
- "a cat runs"
- "the dog barks"
- "an anteater eats"

It can be assumed that the same kind of chunk should operate over all three of these things, however
these two instances are, on paper, radically different due to their differences in discrete symbols.
In order to operate over the similarity of discrete symbol, we can build a separate hierarchy purely
for word-vocabulary to observe semantic meaning in a way that allows it to be a semantic feature for
us to chunk over. 

A broader point here: the point of chunks is to compress similarly stored information and I'm not
sure that building chunks from nothing is enough to simultaneously ascribe meaning to those "nothings".
It's important that we provide semantic meaning as a scaffold to be able to extract structural meaning!

TODO do we do this for all chunks or just the words? I think it's fine to do this for words because chunks
are being stored by their path information but there may come a time where we want to store individual content
instances

TODO another big question is how we extrapolate meaning from words because they're all interdependent on
each other. Do we not need to allow the paths to "converge" to some stable state based on them all being
semantically tagged at the same time? ANSWER LOOK AT SECTION 3.2 OF THE INITIAL COBWEB LANGUAGE PAPER
*   This may also become relevant within the context of chunking - we want to let our chunks' place in the tree converge

TODO thinking about this in more depth, we probably do want to consider having everything in the same hierarchy
so that we can use the same root node concept for 
"""

from cobweb.cobweb_discrete import CobwebTree

class ChunkVocabulary(object):

    """
    A vocabulary for a given corpus that, given a word, will attempt to retrieve the path that
    best explains it!

    Word senses will be categorized according to a single 'content' element and a set of 'context'
    elements and we can use similar path senses to 
    """

    def __init__(self, corpus=[], context_length=3):
        
        self.corpus = corpus
        self.context_length = context_length

        self.tree = CobwebTree(1e-4, True, 0, True, True)

        self.initialize()

    def initialize(self):
        """
        Helper function to prepare all data-structures!
        """
        self.id_to_value = ["EMPTYNULL"]
        for x in self.corpus:
            self.id_to_value.append(x)
        self.value_to_id = dict([(w, i) for i, w in enumerate(self.id_to_value)])
        self.id_count = len(self.corpus)

    def process_window(self, window):
        """
        Primary training function for ChunkVocabulary class - takes in a window and
        aggregates word senses into the vocabulary hierarchy
        """
        pass

    def _add_instance(self, inst):
        """
        
        """
        pass

    def get(self, inst):
        """
        Primary inference function for ChunkVocabulary class - given an input instance, it
        marks the best existing instance in the Cobweb class and identifies the path information
        for such a 
        """
        pass