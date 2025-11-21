"""
A general thresholding class that saves statistics incrementally for the express
purpose of generating chunks.

The prevalent reason of this is that in addition to scoring metrics showcasing the
likelihood that a thing is a chunk, we need a statistic that gradually aggregates
over time and validates the CONTENT specifically. This is inspired by Sequitur, which
merely detects the repetition of a chunk 

We'll create a generic class that can do this and then expand on it through different
statistics that we aggregate.

Important questions:
*   Should all chunks be "thresholded" or should only the first set of extremely primitive
    chunks be thresholded before being added? It shouldn't be this hard to do this but is it
    worth it?
    *   All chunks should be thresholded methinks
*   Should we incorporate this score directly into our scoring metric OR create a boolean
    threshold with which to add chunks?
    *   Partial to either option

NOTE: THIS APPROACH ISN'T ENOUGH! 
"""
import numpy as np

class ChunkThresholder(object):

    """
    Generic Chunk Thresholding class - stores each digram as well as relevant statistics
    to compute heuristics for the "chunk-ness" of that digram to encode it as a valid parse tree.

    We'll be storing data structures which encode the statistics of this class, which update over time,
    as well as providing a set of options that denote heuristics for the creation of a "chunk".
    """
    def __init__(self, id_to_value: list, value_to_id: dict, strategy: str="freq"):
        self.strategy = strategy
        self.id_to_value = id_to_value
        self.value_to_id = value_to_id

        # structures
        self.n_digrams = 0
        self.digram_frequency = np.zeros((len(id_to_value), len(id_to_value)))
        self.token_frequency = np.zeros((len(id_to_value),))

    def aggregate(self, sentence: str, update_stats: bool=True):
        """
        Computes statistics for the given sentence and whether nodes are to be classified
        as chunks or not.

        The workflow is as follows:
        *   Each digram is parsed purely by content
        *   Each digram's heuristic is evaluated
        *   The sentence is added to any relevant data structures
        *   The heuristics are returned (for a sentence with N tokens, N - 1 heuristics)

        We can then use the heuristics to generate necessary chunks depending on some threshold
        within our FiniteParseTree.

        We can pass in the sentence as well as the "update_stats" parameter, which if True adds
        the sentence as data to the statistic analysis.
        """

        tokens = sentence.split(" ")
        heuristics = []
        
        for i in range(len(tokens) - 1):

            idx1 = self.value_to_id[tokens[i]]
            idx2 = self.value_to_id[tokens[i + 1]]

            if update_stats:
                self.n_digrams += 1
                self.digram_frequency[idx1][idx2] += 1
                self.token_frequency[idx1] += 1
                self.token_frequency[idx2] += 1

            h = self._digram_heuristic(tokens[i], tokens[i + 1])
            heuristics.append(h)

        return heuristics

    def _digram_heuristic(self, token1, token2):
        """
        Based on self.strategy, calculates and returns a heuristic for a given digram.

        Some potential statistics we can select from: (check ChatGPT thread) In general, most
        statistics can be approximated by probabilities through an information-theoretic perspective
        OR general frequency. 
        *   General chunk frequency - just storing digram counts
        *   Pointwise Mutual Information - some metric of probability of a given token appearing
            next to another given token
        *   Surprisal Drop - another metric of probability denoting some measure of the given digram's
        """

        idx1 = self.value_to_id[token1]
        idx2 = self.value_to_id[token2]

        digram_freq = self.digram_frequency[idx1][idx2]

        P_t1 = self.token_frequency[idx1] / (self.n_digrams + 1)
        P_t2 = self.token_frequency[idx2] / (self.n_digrams + 1)
        P_digram = digram_freq / self.n_digrams
        pmi = np.log(P_digram / (P_t1 * P_t2))

        
        if self.strategy == "freq":
            return digram_freq
        
        elif self.strategy == "pmi":
            return pmi
        
        elif self.strategy == "merge":
            return (digram_freq, pmi)

        else:
            raise Exception(f"{self.strategy} is not a valid strategy for merge passed into constructor!")
        
