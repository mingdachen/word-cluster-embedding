import mixture_utils
import data_utils
import argparse
import logging
import pickle
import time
import os

import tensorflow as tf
import numpy as np

from models import mixture_classifier


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser(
        description='TensorFlow implementation of \
        \"Smaller Text Classifiers with Discriminative Cluster Embeddings\"')
    parser.register('type', 'bool', str2bool)
    # Basics
    parser.add_argument('--debug', type="bool", default=False,
                        help='whether to activate debug mode (default: False)')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed (default: 0)')
    # Data file
    parser.add_argument('--dataset', type=str, default=None,
                        help='Types of dataset:  yelp-1, yelp-2, \
                        DBPedia, AGNews, IMDB (default: None)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Data path (default: None)')
    parser.add_argument('--vocab_file', type=str, default=None,
                        help='vocab file path (default: None)')
    parser.add_argument('--embed_file', type=str, default=None,
                        help='embedding file path (default: None)')
    # model detail
    parser.add_argument('--embed_dim', type=int, default=100,
                        help='embedding dimension (default: 100)')
    parser.add_argument('--n_embed', type=int, default=10,
                        help='number of embedding vector (default: 10)')
    parser.add_argument('--hidden_size', type=int, default=50,
                        help='hidden dimension of RNN (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--opt', type=str, default='adam',
                        help='types of optimizer: adam (default), \
                        sgd, rmsprop')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                        help='types of optimizer: lstm (default), \
                        gru, rnn')
    parser.add_argument('--bidir', type="bool", default=False,
                        help='whether to use bidirectional \
                        (default: False)')
    # train detail
    parser.add_argument('--save_vocab', type="bool", default=True,
                        help='whether to save vocabulary \
                        (default: True)')
    parser.add_argument('--train_emb', type="bool", default=True,
                        help='whether to train embedding vectors \
                        (default: True)')
    parser.add_argument('--n_epoch', type=int, default=10,
                        help='number of epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch size (default: 20)')
    parser.add_argument('--max_words', type=int, default=50000,
                        help='maximum number of words in vocabulary \
                        (default: 50000)')
    parser.add_argument('--keep_word', type=int, default=1000,
                        help='number of words that use standard embeddings in vocabulary \
                        (default: 1000)')
    parser.add_argument('--proj', type=str, default="gumbel",
                        help='types of embedding projection: gumbel (default)')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='gradient clipping (default: 10)')
    parser.add_argument('--init_temp', type=float, default=0.9,
                        help='initial temperature for gumbel softmax \
                        (default: 0.9)')
    parser.add_argument('--anneal_rate', type=float, default=0.,
                        help='annealing rate for temperature (default: 0)')
    parser.add_argument('--min_temp', type=float, default=0.9,
                        help='minimum temperature (default: 0.9)')
    parser.add_argument('--l2', type=float, default=0.,
                        help='l2 regularizer (default: 0)')
    # misc
    parser.add_argument('--print_every', type=int, default=500,
                        help='print training details after \
                        this number of iterations (default: 500)')
    parser.add_argument('--eval_every', type=int, default=5000,
                        help='evaluate model after \
                        this number of iterations (default: 5000)')
    return parser.parse_args()


def run(args):
    dp = os.path.join(args.data_path, args.dataset.lower() + ".pkl")
    logging.info("loading data from {} ...".format(dp))
    with open(dp, "rb+") as infile:
        revs, vocabs = pickle.load(infile)
    # make vocab
    assert args.proj.lower() == "gumbel", "only gumbel is supported!"
    if not os.path.isfile(args.vocab_file):
        vocab, vocab_mask = mixture_utils.make_dict(
            vocabs.get("train"), args.vocab_file, args.keep_word,
            args.max_words, args.save_vocab)
    else:
        vocab, vocab_mask = mixture_utils.load_dict(args.vocab_file)

    train_data, train_label = data_utils.prepare_data(revs["train"], vocab)
    test_data, test_label = data_utils.prepare_data(revs["test"], vocab)
    dev_data, dev_label = data_utils.prepare_data(revs["dev"], vocab)

    logging.info("#training data: {}".format(len(train_data)))
    logging.info("#dev data: {}".format(len(dev_data)))
    logging.info("#test data: {}".format(len(test_data)))

    logging.info("#unk words in train data: {}".format(
        data_utils.cal_unk(train_data)))
    logging.info("#unk words in dev data: {}".format(
        data_utils.cal_unk(dev_data)))
    logging.info("#unk words in test data: {}".format(
        data_utils.cal_unk(test_data)))

    logging.info("initializing model ...")
    model = mixture_classifier(
        vocab_size=len(vocab),
        vocab_mask=vocab_mask,
        args=args)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables())
    logging.info("model successfully initialized")

    test_d, test_l = data_utils.make_batch(
        test_data, test_label, args.batch_size)
    dev_d, dev_l = data_utils.make_batch(
        dev_data, dev_label, args.batch_size)
    logging.info("-" * 50)

    # training phase
    it = best_dev_pred = 0.
    logging.info("Training start ...")
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(args.n_epoch):
            train_d, train_l = data_utils.make_batch(
                train_data, train_label, args.batch_size)
            loss = n_example = 0
            start_time = time.time()
            for train_doc_, train_label_ in zip(train_d, train_l):
                train_doc_, train_mask_ = data_utils.pad_seq(train_doc_)

                if it % 500 == 0:
                    temp = np.maximum(
                        args.init_temp * np.exp(-args.anneal_rate * it),
                        args.min_temp)

                loss_ = model.train(
                    sess, train_doc_, train_mask_, train_label_, temp)
                loss += loss_ * len(train_doc_)
                n_example += len(train_doc_)
                it += 1

                if it % args.print_every == 0:
                    logging.info("epoch: {}, it: {} (max: {}), "
                                 "loss: {:.5f}, "
                                 "time: {:.5f}(s), temp: {:.5f}"
                                 .format(epoch, it, len(train_d),
                                         loss / n_example,
                                         time.time() - start_time,
                                         temp))
                    loss = n_example = 0
                    start_time = time.time()

                if it % args.eval_every == 0 or it % len(train_d) == 0:
                    start_time = time.time()
                    pred = 0
                    n_dev = 0
                    for dev_doc_, dev_label_ in zip(dev_d, dev_l):
                        dev_doc_, dev_mask_ = data_utils.pad_seq(dev_doc_)
                        pred_ = model.evaluate(
                            sess, dev_doc_, dev_mask_, dev_label_, temp)
                        pred += pred_
                        n_dev += len(dev_doc_)
                    pred /= n_dev
                    logging.info("Dev acc: {:.5f}, #pred: {}, "
                                 "elapsed time: {:.5f}(s)"
                                 .format(pred, n_dev,
                                         time.time() - start_time))
                    start_time = time.time()
                    n_example = 0

                    if best_dev_pred < pred:
                        best_dev_pred = pred
                        best_temp = temp
                        model.save(sess, saver, args.save_dir)
                        logging.info("Best dev acc: {:.5f}"
                                     .format(best_dev_pred))
                        start_time = time.time()

            logging.info("-" * 50)
            start_time = time.time()
            test_temp = temp

            if epoch == args.n_epoch - 1:
                logging.info("final testing ...")
                model.restore(sess, args.save_dir)
                test_temp = best_temp
            pred = n_test = 0

            for test_doc_, test_label_ in zip(test_d, test_l):
                test_doc_, test_mask_ = data_utils.pad_seq(test_doc_)
                pred_ = model.evaluate(
                    sess, test_doc_, test_mask_, test_label_, test_temp)
                pred += pred_
                n_test += len(test_doc_)
            pred /= n_test

            logging.info("-" * 50)
            logging.info("test acc: {:.5f}, #pred: {}, elapsed time: {:.5f}(s)"
                         .format(pred, n_test, time.time() - start_time))
            logging.info("best dev acc: {:.5f}, best temp: {}"
                         .format(best_dev_pred, best_temp))
            logging.info("vocab size: {}".format(len(vocab)))


if __name__ == '__main__':
    args = get_args()
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)
    args.save_dir = "experiments" + "/" + args.dataset.lower() + "/" \
        + args.opt.lower() + "_mix" + str(args.proj) \
        + "_edim" + str(args.embed_dim) \
        + "_nembed" + str(args.n_embed) + "_temb" + str(args.train_emb) \
        + "_hsize" + str(args.hidden_size) \
        + "_itemp" + str(args.init_temp) + "_min_temp" + str(args.min_temp) \
        + "_anneal" + str(args.anneal_rate) + "_lr" + str(args.learning_rate) \
        + "_epoch" + str(args.n_epoch) + "_l2" + str(args.l2) \
        + "_kw" + str(args.keep_word) + "_words" + str(args.max_words)
    args.n_class = data_utils.get_n_class(args.dataset.lower())
    if args.debug:
        args.save_dir = "./mix_exp"
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            datefmt='%m-%d %H:%M')
    else:
        log_file = os.path.join(args.save_dir, 'log')
        print("log saving to", log_file)
        logging.basicConfig(filename=log_file,
                            filemode='w+', level=logging.INFO,
                            format='%(asctime)s %(message)s',
                            datefmt='%m-%d %H:%M')
    if args.dataset is None:
        raise ValueError('dataset is not specified.')
    if args.vocab_file is None:
        raise ValueError('vocab_file is not specified.')
    logging.info(args)
    run(args)
