import logging
import pickle

import numpy as np


def make_dict(
        vocab, vocab_file, n_word_mask, max_words=None, save_vocab=False):
    if max_words is None:
        max_words = len(vocab)
    ls = vocab.most_common(max_words)

    logging.info('#Words: %d -> %d' % (len(vocab), len(ls)))
    for key in ls[:5]:
        logging.info(key)
    logging.info('...')
    for key in ls[-5:]:
        logging.info(key)

    word_to_keep = vocab.most_common(n_word_mask)
    word_to_keep_vocab = [w[0] for w in word_to_keep]
    logging.info("#unique embedding vectors: {}".format(len(word_to_keep)))

    vocab = {w[0]: index + 1 for (index, w) in enumerate(ls)}
    vocab["<unk>"] = 0
    mask = np.zeros(len(vocab)).astype("float32")
    for w in word_to_keep_vocab:
        mask[vocab[w]] = 1.
    if save_vocab:
        logging.info("vocab saving to {}".format(vocab_file))
        with open(vocab_file, "wb+") as vocab_fp:
            pickle.dump([vocab, mask], vocab_fp, protocol=-1)
    return vocab, mask


def load_dict(vocab_file):
    logging.info("loading vocabularies from " + vocab_file + " ...")
    with open(vocab_file, "rb") as vf:
        vocab, mask = pickle.load(vf)
    logging.info("#unique embedding vectors: {}".format(mask.sum()))
    logging.info("vocab size: {}".format(len(vocab)))
    return vocab, mask
