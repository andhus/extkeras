from __future__ import print_function, division

import numpy as np


def greedy(
    predict_f,
    initial_sequence,
    initial_states=None,
    end_criteria=None,
    max_num_steps=1000
):
    """Select the next token highest predicted likelihood iteratively.

    # Arguments
        predict_f (f: (initial_sequence, additional_inputs) -> next_sequence, next_aux_inputs
    """
    if not callable(end_criteria):
        end_value = end_criteria

        def end_criteria(seq):
            return seq[0, -1] == end_value

    t = 0
    cumlogl = 0
    sequence = initial_sequence
    states = initial_states if initial_states is not None else []
    while not end_criteria(sequence) and t < max_num_steps:
        label_proba, states = predict_f(sequence, states)
        t += 1
        next_element = np.argmax(label_proba, axis=-1)
        cumlogl += np.log(label_proba[..., next_element])
        sequence = np.concatenate([sequence, next_element[None]], axis=-1)

    return sequence, cumlogl


def beam_search(input_text,
                          search_width=20,
                          branch_factor=None,
                          t_max=None):
    """Perform beam search to (approximately) find the translated sentence that
    maximises the conditional probability given the input sequence.

    Returns the completed sentences (reached end-token) in order of decreasing
    score (the first is most probable) followed by incomplete sentences in order
    of decreasing score - as well as the score for the respective sentence.

    References:
        [1] "Sequence to sequence learning with neural networks"
        (https://arxiv.org/pdf/1409.3215.pdf)
    """

    if branch_factor is None:
        branch_factor = search_width
    elif branch_factor > search_width:
        raise ValueError("branch_factor must be smaller than search_width")
    elif branch_factor < 2:
        raise ValueError("branch_factor must be >= 2")

    def k_largest_val_idx(a, k):
        """Returns top k largest values of a and their indices, ordered by
        decreasing value"""
        top_k = np.argpartition(a, -k)[-k:]
        return sorted(zip(a[top_k], top_k))[::-1]

    class Beam(object):
        """A Beam holds the tokens seen so far and its accumulated score
        (log likelihood).
        """

        def __init__(self, sequence, scores=(0,)):
            self._sequence = list(sequence)
            self.scores = list(scores)

        @property
        def sequence(self):
            return np.array(self._sequence)

        @property
        def score(self):
            return sum(self.scores)

        def get_child(self, element, score):
            return Beam(self._sequence + [element], self.scores + [score])

        def __len__(self):
            return len(self._sequence)

        def __lt__(self, other):
            return self.score < other.score

        def __gt__(self, other):
            return other.score > other.score

    x_ = np.array(input_tokenizer.texts_to_sequences([input_text]))
    x_ = np.repeat(x_, search_width, axis=0)

    if t_max is None:
        t_max = x_.shape[-1] * 2

    start_idx = target_tokenizer.word_index[start_token]
    end_idx = target_tokenizer.word_index[end_token]

    # A list of the <search_width> number of beams with highest score is
    # maintained through out the search. Initially there is only one beam.
    incomplete_beams = [Beam(sequence=[start_idx], scores=[0])]
    # All beams that reached the end-token are kept separately.
    complete_beams = []

    t = 0
    while len(complete_beams) < search_width and t < t_max:
        t += 1
        # create a batch of inputs representing the incomplete_beams
        y_ = np.vstack([beam.sequence for beam in incomplete_beams])
        # predict next tokes for every incomplete beam
        batch_size = len(incomplete_beams)
        y_pred_ = model.predict([y_, x_[:batch_size]])
        # from each previous beam create new candidate beams and save the once
        # with highest score for next iteration.
        beams_updated = []
        for i, beam in enumerate(incomplete_beams):
            for proba, idx in k_largest_val_idx(y_pred_[i, -1], branch_factor):
                new_beam = beam.get_child(element=idx, score=np.log(proba))
                if idx == end_idx:
                    # beam completed
                    complete_beams.append(new_beam)
                elif len(beams_updated) < search_width:
                    # not full search width
                    heapq.heappush(beams_updated, new_beam)
                elif new_beam.score > beams_updated[0].score:
                    # better than candidate with lowest score
                    heapq.heapreplace(beams_updated, new_beam)
                else:
                    # if score is worse that existing candidates we abort search
                    # for children of this beam (since next token processed
                    # in order of decreasing likelihood)
                    break

        # faster to process beams in order of decreasing score next iteration,
        # due to break above
        incomplete_beams = sorted(beams_updated, reverse=True)

    # want to return in order of decreasing score
    complete_beams = sorted(complete_beams, reverse=True)

    output_texts = []
    output_scores = []
    for beam in complete_beams + incomplete_beams:
        text = target_tokenizer.sequences_to_texts(beam.sequence[None])[0]
        output_texts.append(text)
        # average score, skipping start token
        output_scores.append(beam.score / (len(beam) - 1))

    return output_texts, output_scores
