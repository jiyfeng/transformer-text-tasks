# In NMT, new sentences are translated by a simple beam search decoder that finds a translation that
# approximately maximizes the conditional probability of a trained NMT model. The beam search strategy
# generates the translation word by word from left-to-right while keeping a fixed number (beam) of
# active candidates at each time step. By increasing the beam size, the translation performance can
# increase at the expense of significantly reducing the decoder speed.

# To avoid underflowing the floating point numbers, the natural logarithm of the probabilities are
# multiplied together, which keeps the numbers larger and manageable. Further, it is also common to
# perform the search by minimizing the score, therefore, the negative log of the probabilities are
# multiplied. This final tweak means that we can sort all candidate sequences in ascending order by
# their score and select the first k as the most likely candidate sequences.

# Beam Search Strategies for Neural Machine Translation - https://arxiv.org/abs/1702.01806
import numpy as np
from math import log

def beam_search(data, k):
    sequences = [[list(), 1]]
    for row in data:
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = (seq + [j], score * -log(row[j]))
                all_candidates.append(candidate)
        sequences = sorted(all_candidates, key=lambda t: t[1]) [:k]
    return sequences

if __name__ == '__main__':
    data = [[0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1]]
    data = np.array(data)
    # decode sequence
    result = beam_search(data, 3)
    # print result
    for seq in result:
        print(seq)