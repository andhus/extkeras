from __future__ import print_function, division

import numpy as np
from numpy.testing.utils import assert_equal

from extkeras.readout import greedy


def test_greedy():
    num_classes = 5

    def predict(sequence, states=None):
        prev_class = sequence[0, -1]
        proba = np.zeros((1, num_classes))
        proba[0, (prev_class + 1) % num_classes] = 1.

        return proba, []

    final_sequence, cumlogl = greedy(
        predict,
        initial_sequence=np.array([[0]]),
        max_num_steps=4
    )

    sequence_expected = np.arange(5)[None, :]
    assert_equal(final_sequence, sequence_expected)
