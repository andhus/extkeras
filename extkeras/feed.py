from __future__ import print_function

from collections import namedtuple

import numpy as np


def local_batch_shuffle(idx_batches, p=0.5):
    for batch, next_batch in zip(idx_batches[:-1], idx_batches[1:]):
        np.random.shuffle(batch)
        np.random.shuffle(next_batch)
        num_moved = 0
        for i in range(len(batch)):
            if num_moved < len(next_batch):
                if np.random.random() < p:
                    replace_idx = next_batch[num_moved]
                    next_batch[num_moved] = batch[i]
                    batch[i] = replace_idx
                    num_moved += 1
            else:
                break


def bidirectional_local_batch_shuffle(idx_batches, p=0.5):
    local_batch_shuffle(idx_batches, p)
    local_batch_shuffle(idx_batches[::-1], p)


def get_balanced_idx_batches(sequence_lengths, max_elements):
    idx_batch = []
    idx_batches = [idx_batch]
    batch_size = 0
    for length, idx in sorted(
        zip(sequence_lengths, range(len(sequence_lengths)))
    ):
        if batch_size + length <= max_elements:
            idx_batch.append(idx)
            batch_size += length
        else:
            idx_batch = [idx]
            idx_batches.append(idx_batch)
            batch_size = length

    return idx_batches


class SequenceLengthBatching(object):

    def __init__(
        self,
        sequences,
        batch_size,
        cap_length=None,
        shuffle_p=0.5,
        postprocess_f=None
    ):
        self.sequences = sequences
        self.batch_size = batch_size
        if postprocess_f is not None:
            self.postprocess_f = postprocess_f
        else:
            self.postprocess_f = lambda x: x
        self.cap_length = cap_length
        self.shuffle_p = shuffle_p

        self.lengths = [len(s) for s in sequences]
        if self.cap_length:
            self.lengths_caped = [min(len(s), cap_length) for s in sequences]
        else:
            self.lengths_caped = self.lengths

        self.max_length = max(self.lengths)
        self.max_length_caped = max(self.lengths_caped)

        self.average_length = sum(self.lengths) / len(self.sequences)
        self.average_length_caped = sum(self.lengths_caped) / len(self.sequences)

        self.number_of_elements_caped = sum(self.lengths_caped)
        self.max_num_elements_per_batch = int(
            self.batch_size * self.average_length_caped
        )
        self.balanced_idx_batches = get_balanced_idx_batches(
            sequence_lengths=self.lengths_caped,
            max_elements=self.max_num_elements_per_batch
        )
        self.num_batches = len(self.balanced_idx_batches)

    def get_generator(self):
        idx_batches = get_balanced_idx_batches(
            sequence_lengths=self.lengths_caped,
            max_elements=self.max_num_elements_per_batch
        )
        local_batch_shuffle(idx_batches, p=self.shuffle_p)
        for idx_batch in idx_batches:
            sequence_batch = [
                self.sequences[idx][:self.max_length] for idx in idx_batch
            ]
            yield self.postprocess_f(sequence_batch)


class SequenceTuple(object):

    def __init__(self, elements):
        self.elements = elements

    def __len__(self):
        return sum([len(e) for e in self.elements])


def postprocess()