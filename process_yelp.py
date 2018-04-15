import argparse
import logging
import pickle
import re
import os

from sklearn.model_selection import train_test_split
from collections import Counter


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser(
        description='Data Processing for \
        \"Smaller Text Classifiers with Discriminative Cluster Embeddings\"')
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--type', type=str, default="yelp-1",
                        help='data type: yelp-1, yelp-2')
    parser.add_argument('--path', type=str, default=None,
                        help='data path')
    parser.add_argument('--clean_str', type="bool", default=True,
                        help='whether to tokenize data')
    args = parser.parse_args()
    return args


def load_split_data(data_file, dataset, clean_str):
    logging.info("loading data from {} ...".format(data_file))
    revs = []
    vocab = Counter()
    with open(data_file, "rb") as f:
        for line in f:
            label_, data_ = \
                line.decode('unicode_escape').strip("\n").split(",", 1)
            label_ = int(label_.strip("\"")) - 1
            if clean_str:
                rev = []
                rev.append(data_.strip(r'\"'))
                data_ = clean_string(" ".join(rev))
            else:
                data_ = data_.strip(r'\"')
            for i, word in enumerate(data_.split(" ")):
                vocab[word] += 1
            datum = {"y": label_,
                     "text": data_.split(" "),
                     "num_words": len(data_.split(" "))}
            revs.append(datum)
    return revs, vocab


def load_data(path, data_folder, dataset, clean_str):
    """
    Loads data.
    """
    revs = {}
    vocabs = {}
    train_file = os.path.join(path, data_folder[0])
    if data_folder[1] is not None:
        dev_file = os.path.join(path, data_folder[1])
    else:
        dev_file = None
    test_file = os.path.join(path, data_folder[2])

    revs_split, vocab_split = load_split_data(train_file, dataset, clean_str)
    revs["train"] = revs_split
    vocabs["train"] = vocab_split
    if dev_file is None:
        train_split, test_split = \
            train_test_split(revs_split, test_size=5000)
        word_count = Counter()
        for data in test_split:
            for word in data["text"]:
                word_count[word] += 1
        vocabs["train"] = vocabs["train"] - word_count
        revs["train"] = train_split

        vocabs["dev"] = word_count
        revs["dev"] = test_split
    else:
        revs_split, vocab_split = \
            load_split_data(dev_file, dataset, clean_str)
        revs["dev"] = revs_split
        vocabs["dev"] = vocab_split
    revs_split, vocab_split = load_split_data(test_file, dataset, clean_str)
    revs["test"] = revs_split
    vocabs["test"] = vocab_split
    return revs, vocabs


def clean_string(string):
    """
    Tokenization/string cleaning for yelp data set
    Based on https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\"\"", " \" ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M')
    args = get_args()
    # train, dev, test
    if args.type.lower() == "yelp-1":
        data_folder = ["train.csv",
                       None,
                       "test.csv"]
    elif args.type.lower() == "yelp-2":
        data_folder = ["train.csv",
                       None,
                       "test.csv"]
    else:
        raise ValueError("invalid dataset type: {}".format(args.type))
    revs, vocab = \
        load_data(args.path, data_folder, args.type, args.clean_str)
    logging.info("data loaded!")
    for split in ["train", "dev", "test"]:
        if revs.get(split) is not None:
            logging.info(split + " " + "-" * 50)
            logging.info("number of sentences: " + str(len(revs[split])))
            logging.info("vocab size: {}".format(len(vocab[split])))
    logging.info("-" * 50)
    logging.info("total vocab size: {}".format(
        len(sum(vocab.values(), Counter()))))
    logging.info("total data size: {}".format(len(sum(revs.values(), []))))
    pickle.dump([revs, vocab],
                open(args.type.lower() + ".pkl", "wb+"),
                protocol=-1)
    logging.info("dataset created!")
