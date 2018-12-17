from __future__ import print_function, division

from nose.tools import assert_equal


from extkeras.feed import get_balanced_idx_batches


def test_get_balanced_idx_batches():
    from extkeras.feed import get_balanced_idx_batches

    batches = get_balanced_idx_batches(
        sequence_lengths=range(1, 11),
        max_elements=11
    )
    batches_expected = [
        [0, 1, 2, 3],
        [4, 5],
        [6],
        [7],
        [8],
        [9],
    ]
    assert_equal(batches, batches_expected)


class TestSequenceLengthBatching(object):
    from extkeras.feed import SequenceLengthBatching as test_class

    def test_balanced_batches(self):
        pass