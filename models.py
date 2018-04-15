import logging
import os

import tensorflow as tf

from model_utils import gumbel_softmax, softmax_with_temperature


class base_classifier:
    def __init__(self, args):
        # model configurations
        self.hidden_size = args.hidden_size
        self.embed_dim = args.embed_dim
        self.n_embed = args.n_embed
        self.n_class = args.n_class
        self.bidir = args.bidir
        self.proj = args.proj

        # training configurations
        self.trian_emb = args.train_emb
        self.grad_clip = args.grad_clip
        self.lr = args.learning_rate
        self.opt = args.opt
        self.l2 = args.l2

        if args.rnn_type.lower() == 'lstm':
            self.cell = tf.contrib.rnn.BasicLSTMCell
        elif args.rnn_type.lower() == 'gru':
            self.cell = tf.contrib.rnn.GRUCell
        elif args.rnn_type.lower() == 'rnn':
            self.cell = tf.contrib.rnn.core_rnn_cell.BasicRNNCell
        else:
            raise NotImplementedError('Invalid rnn type: %s' % args.rnn_type)

    def _build_graph(self):
        raise NotImplementedError()

    def _build_optimizer(self, loss):
        if self.opt.lower() == 'sgd':
            opt = tf.train.GradientDescentOptimizer(self.lr)
        elif self.opt.lower() == 'adam':
            opt = tf.train.AdamOptimizer(self.lr)
        elif self.opt.lower() == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(self.lr)
        else:
            raise NotImplementedError("Invalid type of optimizer: {}"
                                      .format(self.opt))

        vars_list = tf.trainable_variables()
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2)
        reg_term = tf.contrib.layers.apply_regularization(
            regularizer,
            [v for v in vars_list if v.name.split(":")[0] != "gs_param"])
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss + reg_term, vars_list), self.grad_clip)
        self.updates = opt.apply_gradients(zip(grads, vars_list))

    def train(self, sess, inputs, mask, labels, temp):
        feed_dict = {self.inputs: inputs, self.mask: mask,
                     self.tau: temp, self.labels: labels}
        _, loss = sess.run(
            [self.updates, self.loss], feed_dict)
        return loss

    def evaluate(self, sess, inputs, mask, labels, temp):
        feed_dict = {self.inputs: inputs, self.mask: mask,
                     self.tau: temp, self.labels: labels}
        acc = sess.run(self.acc, feed_dict)
        return acc

    def save(self, sess, saver, save_dir):
        save_path = os.path.join(save_dir, 'model.ckpt')
        saver.save(sess, save_path)
        logging.info("model saved to {}".format(save_path))

    def restore(self, sess, save_dir):
        """
        restore model
        """
        save_path = os.path.join(save_dir, 'model.ckpt')
        loader = tf.train.import_meta_graph(save_path + '.meta')
        loader.restore(sess, save_path)
        logging.info("model restored from {}".format(save_path))


class sentiment_classifier(base_classifier):
    def __init__(self, vocab_size, args):
        super(sentiment_classifier, self).__init__(args)
        self.vocab_size = vocab_size
        self.aux_dim = args.aux_dim
        self._build_graph()
        self._build_optimizer(self.loss)

    def _build_graph(self):
        self.tau = tf.placeholder(tf.float32, name="temperature")
        self.inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
        self.labels = tf.placeholder(tf.int32, [None, ], name="labels")
        self.mask = tf.placeholder(tf.int32, [None, None], name="mask")

        if self.proj.lower() == "gumbel":
            self.embedding = tf.get_variable(
                "embedding", [self.n_embed, self.embed_dim],
                trainable=self.trian_emb,
                initializer=tf.random_uniform_initializer(
                    minval=-0.1, maxval=0.1))
            self.gs_param = tf.get_variable(
                "gs_param", [self.vocab_size, self.n_embed],
                initializer=tf.random_uniform_initializer(maxval=1))

            logits = tf.nn.embedding_lookup(self.gs_param, self.inputs)
            batch_size, batch_len = \
                tf.shape(self.inputs)[0], tf.shape(self.inputs)[1]
            embed_prob = gumbel_softmax(
                tf.reshape(logits, [batch_size * batch_len, -1]),
                self.tau, hard=False)

            inputs_embed = tf.matmul(embed_prob, self.embedding)
            inputs_embed = tf.reshape(
                inputs_embed, [batch_size, batch_len, self.embed_dim])

            test_embed_prob = softmax_with_temperature(
                tf.reshape(logits, [batch_size * batch_len, -1]),
                self.tau, hard=True)
            test_inputs_embed = tf.matmul(
                test_embed_prob, self.embedding)
            test_inputs_embed = tf.reshape(
                inputs_embed, [batch_size, batch_len, self.embed_dim])

        elif self.proj.lower() == "standard":
            self.embedding = tf.get_variable(
                "embedding", [self.vocab_size, self.embed_dim],
                trainable=self.trian_emb,
                initializer=tf.random_uniform_initializer(
                    minval=-0.1, maxval=0.1))
            inputs_embed = tf.nn.embedding_lookup(self.embedding, self.inputs)
            test_inputs_embed = inputs_embed
        else:
            raise NotImplementedError(
                "invalid projection type: {}".format(self.proj))

        if self.aux_dim:
            self.aux_embedding = tf.get_variable(
                "aux_embedding", [self.vocab_size, self.aux_dim],
                trainable=self.trian_emb,
                initializer=tf.random_uniform_initializer(
                    minval=-0.1, maxval=0.1))
            aux_inputs_emb = \
                tf.nn.embedding_lookup(self.aux_embedding, self.inputs)

            inputs_embed = tf.concat([inputs_embed, aux_inputs_emb], axis=-1)
            test_inputs_embed = tf.concat(
                [test_inputs_embed, aux_inputs_emb], axis=-1)

        cell = self.cell(self.hidden_size)
        seq_length = tf.reduce_sum(self.mask, axis=1)

        with tf.variable_scope("dynamic_rnn") as scope:
            self.states, final_state = \
                tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=inputs_embed,
                    sequence_length=seq_length,
                    dtype=tf.float32,
                    scope="dynamic_rnn")
            if type(final_state) is tf.nn.rnn_cell.LSTMStateTuple:
                self.final_state = final_state.h
            else:
                self.final_state = final_state
            scope.reuse_variables()
            self.test_states, final_state = \
                tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=test_inputs_embed,
                    sequence_length=seq_length,
                    dtype=tf.float32,
                    scope="dynamic_rnn")
            if type(final_state) is tf.nn.rnn_cell.LSTMStateTuple:
                self.test_final_state = final_state.h
            else:
                self.test_final_state = final_state

        prob = tf.layers.dense(
            self.final_state, self.n_class,
            activation=None, name="prob")
        log_y_given_h = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels,
            logits=prob,
            name="cross_entropy")
        test_prob = tf.layers.dense(
            self.test_final_state, self.n_class,
            activation=None, name="prob", reuse=True)
        self.loss = tf.reduce_mean(log_y_given_h)

        self.pred = tf.cast(tf.argmax(test_prob, axis=1), tf.int32)
        self.acc = tf.reduce_sum(
            tf.cast(tf.equal(self.labels, self.pred), tf.float32))


class mixture_classifier(base_classifier):
    def __init__(self, vocab_size, vocab_mask, args):
        super(mixture_classifier, self).__init__(args)
        self.vocab_size = vocab_size
        self.vocab_mask = vocab_mask

        self._build_graph()
        self._build_optimizer(self.loss)

    def _build_graph(self):
        self.tau = tf.placeholder(tf.float32, name="temperature")
        self.inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
        self.labels = tf.placeholder(tf.int32, [None, ], name="labels")
        self.mask = tf.placeholder(tf.int32, [None, None], name="mask")
        self.vocab_mask = tf.Variable(
            self.vocab_mask, trainable=False, name="vocab_mask")

        self.embedding = tf.get_variable(
            "embedding", [self.vocab_size, self.embed_dim],
            trainable=self.trian_emb,
            initializer=tf.random_uniform_initializer(
                minval=-0.1, maxval=0.1))

        if self.proj.lower() == "gumbel":
            self.cluster_embedding = tf.get_variable(
                "cluster_embedding", [self.n_embed, self.embed_dim],
                trainable=self.trian_emb,
                initializer=tf.random_uniform_initializer(
                    minval=-0.1, maxval=0.1))
            self.gs_param = tf.get_variable(
                "gs_param", [self.vocab_size, self.n_embed],
                initializer=tf.random_uniform_initializer(maxval=1))

            logits = tf.nn.embedding_lookup(self.gs_param, self.inputs)
            batch_size, batch_len = \
                tf.shape(self.inputs)[0], tf.shape(self.inputs)[1]

            embed_prob = gumbel_softmax(
                tf.reshape(logits, [batch_size * batch_len, -1]),
                self.tau, hard=False)
            inputs_embed = tf.matmul(embed_prob, self.cluster_embedding)
            inputs_embed = tf.reshape(
                inputs_embed, [batch_size, batch_len, self.embed_dim])

            test_embed_prob = softmax_with_temperature(
                tf.reshape(logits, [batch_size * batch_len, -1]),
                self.tau, hard=True)
            test_inputs_embed = tf.matmul(
                test_embed_prob, self.cluster_embedding)
            test_inputs_embed = tf.reshape(
                inputs_embed, [batch_size, batch_len, self.embed_dim])
        else:
            raise NotImplementedError(
                "invalid projection type: {}".format(self.proj))

        v_mask = tf.nn.embedding_lookup(self.vocab_mask, self.inputs)
        full_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)

        v_mask = tf.expand_dims(v_mask, axis=-1)

        inputs_embed = full_emb * v_mask + inputs_embed * (1. - v_mask)
        test_inputs_embed = \
            full_emb * v_mask + test_inputs_embed * (1. - v_mask)

        cell = self.cell(self.hidden_size)
        seq_length = tf.reduce_sum(self.mask, axis=1)

        with tf.variable_scope("dynamic_rnn") as scope:
            self.states, final_state = \
                tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=inputs_embed,
                    sequence_length=seq_length,
                    dtype=tf.float32,
                    scope="dynamic_rnn")
            if type(final_state) is tf.nn.rnn_cell.LSTMStateTuple:
                self.final_state = final_state.h
            else:
                self.final_state = final_state

            scope.reuse_variables()
            self.test_states, final_state = \
                tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=test_inputs_embed,
                    sequence_length=seq_length,
                    dtype=tf.float32,
                    scope="dynamic_rnn")
            if type(final_state) is tf.nn.rnn_cell.LSTMStateTuple:
                self.test_final_state = final_state.h
            else:
                self.test_final_state = final_state

        prob = tf.layers.dense(
            self.final_state, self.n_class,
            activation=None, name="prob")
        log_y_given_h = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels,
            logits=prob,
            name="cross_entropy")

        test_prob = tf.layers.dense(
            self.test_final_state, self.n_class,
            activation=None, name="prob", reuse=True)
        self.loss = tf.reduce_mean(log_y_given_h)

        self.pred = tf.cast(tf.argmax(test_prob, axis=1), tf.int32)
        self.acc = tf.reduce_sum(
            tf.cast(tf.equal(self.labels, self.pred), tf.float32))
