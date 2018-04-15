import logging
import pickle

import numpy as np

from collections import Counter


def get_dict(data):
    word_count = Counter()
    for sent in data:
        for word in sent:
            word_count[word] += 1
    return word_count


def get_n_class(dataset):
    if dataset.lower() == "yelp-1":
        return 5
    elif dataset.lower() == "yahoo":
        return 10
    elif dataset.lower() == "agnews":
        return 4
    elif dataset.lower() == "dbpedia":
        return 14
    else:
        return 2


def make_dict(
        vocab, vocab_file, max_words=None, save_vocab=False):
    if max_words is None:
        max_words = len(vocab)

    ls = vocab.most_common(max_words)

    logging.info('#Words: %d -> %d' % (len(vocab), len(ls)))
    for key in ls[:5]:
        logging.info(key)
    logging.info('...')
    for key in ls[-5:]:
        logging.info(key)

    vocab = {w[0]: index + 1 for (index, w) in enumerate(ls)}
    if save_vocab:
        logging.info("vocab saving to {}".format(vocab_file))
        with open(vocab_file, "wb+") as vocab_fp:
            pickle.dump(vocab, vocab_fp, protocol=-1)
    vocab["<unk>"] = 0
    return vocab


def load_dict(vocab_file):
    logging.info("loading vocabularies from " + vocab_file + " ...")
    with open(vocab_file, "rb") as vocab_fp:
        vocab = pickle.load(vocab_fp)
    vocab["<unk>"] = 0
    logging.info("vocab size: {}".format(len(vocab)))
    return vocab


def data_to_idx(data, vocab):
    return [to_idxs(sent, vocab) for sent in data]


def prepare_data(revs, vocab):
    data = []
    label = []
    for rev in revs:
        data.append(to_idxs(rev["text"], vocab))
        label.append(rev["y"])
    return np.asarray(data), np.asarray(label)


def make_batch(revs, labels, batch_size, shuffle=True):
    n = len(revs)
    revs = np.asarray(revs)
    labels = np.asarray(labels)
    if shuffle:
        perm = np.arange(n)
        np.random.shuffle(perm)
        revs = revs[perm]
        labels = labels[perm]
    idx_list = np.arange(0, n, batch_size)
    batch_data = []
    batch_label = []
    for idx in idx_list:
        batch_data.append(
            revs[np.arange(idx, min(idx + batch_size, n))])
        batch_label.append(
            labels[np.arange(idx, min(idx + batch_size, n))])
    return batch_data, batch_label


def pad_seq(data):
    data_len = [len(data_) for data_ in data]
    max_len = np.max(data_len)
    data_holder = np.zeros((len(data), max_len))
    mask = np.zeros_like(data_holder)

    for i, data_ in enumerate(data):
        data_holder[i, :len(data_)] = np.asarray(data_)
        mask[i, :len(data_)] = 1
    return data_holder, mask


def show_data(seqs, inv_dict):
    def inv_vocab(x):
        return inv_dict[x]
    tmp = ""
    for seq in seqs:
        if isinstance(seq, np.int32) or isinstance(seq, int):
            tmp += inv_dict[seq] + ' '
        else:
            print(' '.join(list(map(inv_vocab, seq))))
    if tmp != "":
        print(tmp)


# given the corresponding the char and return the index
def to_index(word, vocab):
    return vocab.get(word, 0)


# given the corresponding the chars and return the indexes.
def to_idxs(words, vocab):
    idxs = [to_index(word, vocab) for word in words]
    return idxs


def cal_unk(sents):
    unk_count = 0
    total_count = 0
    for sent in sents:
        for w in sent:
            if w == 0:
                unk_count += 1
            total_count += 1
    return unk_count, total_count, unk_count / total_count
