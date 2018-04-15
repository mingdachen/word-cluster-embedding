import pickle
import logging
import argparse
import nltk
import os

from collections import Counter
from sklearn.model_selection import train_test_split


MAX_SENT_LEN = 400


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser(
        description='Data Processing for \
        \"Smaller Text Classifiers with Discriminative Cluster Embeddings\"')
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--type', type=str, default="imdb",
                        help='data type: imdb')
    parser.add_argument('--path_train_pos', type=str, default=None,
                        help='positive train data path')
    parser.add_argument('--path_train_neg', type=str, default=None,
                        help='negative train data path')
    parser.add_argument('--path_test_pos', type=str, default=None,
                        help='positive test data path')
    parser.add_argument('--path_test_neg', type=str, default=None,
                        help='negative test data path')
    parser.add_argument('--dev_size', type=float, default=2000,
                        help='dev set size')
    parser.add_argument('--train_size', type=float, default=1.0,
                        help='train set ratio')
    args = parser.parse_args()
    return args


def clean_review(line):
    return nltk.wordpunct_tokenize(line.strip())[: MAX_SENT_LEN]


def process_text_files(file_dir, label):
    logging.info("loading data from {} ...".format(file_dir))
    vocab = Counter()
    revs = []
    for filename in os.listdir(file_dir):
        filepath = os.path.join(file_dir, filename)
        with open(filepath, 'r') as f:
            words = clean_review(f.read().lower())
            for w in words:
                vocab[w] += 1
        datum = {"y": label,
                 "text": words,
                 "num_words": len(words)}
        revs.append(datum)
    return revs, vocab


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M')
    args = get_args()

    train_pos_revs, train_pos_vocab = \
        process_text_files(args.path_train_pos, 1)
    train_neg_revs, train_neg_vocab = \
        process_text_files(args.path_train_neg, 0)
    test_pos_revs, test_pos_vocab = \
        process_text_files(args.path_test_pos, 1)
    test_neg_revs, test_neg_vocab = \
        process_text_files(args.path_test_neg, 0)
    logging.info("data loaded!")

    train_pos_revs, val_pos_revs = \
        train_test_split(
            train_pos_revs, test_size=args.dev_size // 2)
    train_neg_revs, val_neg_revs = \
        train_test_split(
            train_neg_revs, test_size=args.dev_size // 2)

    all_val = val_pos_revs + val_neg_revs

    val_vocab = Counter()
    for d in all_val:
        for w in d["text"]:
            val_vocab[w] += 1
    if args.train_size != 1:
        train_vocab = Counter()
        _, train_pos_revs = \
            train_test_split(
                train_pos_revs,
                test_size=args.train_size)
        for d in train_pos_revs:
            for w in d["text"]:
                train_vocab[w] += 1
        _, train_neg_revs = \
            train_test_split(
                train_neg_revs,
                test_size=args.train_size)
        for d in train_neg_revs:
            for w in d["text"]:
                train_vocab[w] += 1
        vocab = {"train": train_vocab,
                 "dev": val_vocab,
                 "test": test_pos_vocab + test_neg_vocab}
    else:
        vocab = {"train": train_pos_vocab + train_neg_vocab - val_vocab,
                 "dev": val_vocab, "test": test_pos_vocab + test_neg_vocab}

    revs = {"train": train_pos_revs + train_neg_revs,
            "dev": all_val, "test": test_pos_revs + test_neg_revs}

    for split in ["train", "dev", "test"]:
        if revs.get(split) is not None:
            logging.info(split + " " + "-" * 50)
            logging.info("number of sentences: " + str(len(revs[split])))
            logging.info("vocab size: {}".format(len(vocab[split])))
    logging.info("-" * 50)
    logging.info("total vocab size: {}".format(
        len(sum(vocab.values(), Counter()))))
    logging.info("total data size: {}".format(len(sum(revs.values(), []))))
    save_path = args.type.lower() + str(args.train_size) + ".pkl"
    pickle.dump([revs, vocab], open(save_path, "wb+"), protocol=-1)
    logging.info("dataset saved to {}".format(save_path))
