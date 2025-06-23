import torch
import numpy as np


class WordNoising(object):
    """Generate a noisy version of a sentence, without changing words themselves."""
    def __init__(self, dictionary, bpe_cont_marker="@@", bpe_end_marker=None):
        self.dictionary = dictionary
        self.bpe_end = None
        if bpe_cont_marker:
            self.bpe_end = np.array([
                not self.dictionary[i].endswith(bpe_cont_marker)
                for i in range(len(self.dictionary))
            ])
        elif bpe_end_marker:
            self.bpe_end = np.array([
                self.dictionary[i].endswith(bpe_end_marker)
                for i in range(len(self.dictionary))
            ])

        self.get_word_idx = (
            self._get_bpe_word_idx
            if self.bpe_end is not None
            else self._get_token_idx
        )

    def noising(self, x, lengths, noising_prob=0.0):
        raise NotImplementedError()

    def _get_bpe_word_idx(self, x):
        """
        Given a list of BPE tokens, for every index in the tokens list,
        return the index of the word grouping that it belongs to.
        For example, for input x corresponding to ["how", "are", "y@@", "ou"],
        return [[0], [1], [2], [2]].
        """
        # x: (T x B)
        bpe_end = self.bpe_end[x]

        if (x.size(0) == 1 and x.size(1) == 1):
            # Special case when we only have one word in x. If x = [[N]],
            # bpe_end is a scalar (bool) instead of a 2-dim array of bools,
            # which makes the sum operation below fail.
            return np.array([[0]])

        # do a reduce front sum to generate word ids
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx
        return word_idx

    def _get_token_idx(self, x):
        """
        This is to extend noising functions to be able to apply to non-bpe
        tokens, e.g. word or characters.
        """
        x = torch.t(x)
        word_idx = np.array([range(len(x_i)) for x_i in x])
        return np.transpose(word_idx)

class WordShuffle(WordNoising):
    """Shuffle words by no more than k positions."""

    def __init__(self, dictionary, default_max_shuffle_distance=3, bpe_cont_marker="@@", bpe_end_marker=None):
        super().__init__(dictionary, bpe_cont_marker, bpe_end_marker)
        self.default_max_shuffle_distance = 3

    def noising(self, x, lengths, max_shuffle_distance=None):
        if max_shuffle_distance is None:
            max_shuffle_distance = self.default_max_shuffle_distance
        # x: (T x B), lengths: B
        if max_shuffle_distance == 0:
            return x, lengths

        # max_shuffle_distance < 1 will return the same sequence
        assert max_shuffle_distance > 1

        # define noise word scores
        noise = np.random.uniform(
            0,
            max_shuffle_distance,
            size=(x.size(0), x.size(1)),
        )
        noise[0] = -1  # do not move start sentence symbol
        # be sure to shuffle entire words
        word_idx = self.get_word_idx(x)
        x2 = x.clone()
        for i in range(lengths.size(0)):
            length_no_eos = lengths[i]
            if x[lengths[i] - 1, i] == self.dictionary.eos():
                length_no_eos = lengths[i] - 1
            # generate a random permutation
            scores = word_idx[:length_no_eos, i] + noise[word_idx[:length_no_eos, i], i]
            # ensure no reordering inside a word
            scores += 1e-6 * np.arange(length_no_eos)
            permutation = scores.argsort()
            # shuffle words
            x2[:length_no_eos, i].copy_(
                x2[:length_no_eos, i][torch.from_numpy(permutation)]
            )
        return x2, lengths
