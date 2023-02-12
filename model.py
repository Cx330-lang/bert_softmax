# -*- encoding: utf-8 -*-
"""
@ Author: 钱朗
@ File: __init__.py
"""


import os
import re

import tensorflow as tf

from bert import modeling
from bert import optimization


def average_gradients(tower_grads):
    average_grads = []

    for grad_and_vars in zip(*tower_grads):
        grads = []

        for g, _ in grad_and_vars:
            extend_g = tf.expand_dims(g, 0)
            grads.append(extend_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]

        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)

    return average_grads


class BertClassifirer(object):
    def __init__(self, config):
        # self.loss = None
        self.__bert_config_path = os.path.join(config["bert_model_path"], "bert_config.json")
        self.__num_classes = config["num_classes"]
        self.__learning_rate = config["learning_rate"]

        self.__num_train_step = config["train_steps"]
        self.__num_warmup_step = config["warmup_steps"]

        self.__use_lmcl = config["use_lmcl"]

        self._use_focal_loss = config["use_focal_loss"]

        self.__use_r_drop = config["use_r_drop"]
        self.__dropout_rate = config["dropout_rate"]

        self.__use_label_smoothing = config["use_label_smoothing"]
        self.__smooth_rate = config["smooth_rate"]

        self.batch_size = config["batch_size"]
        self.sequence_len = config["sequence_length"]

        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_masks = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids")
        self.label_ids = tf.placeholder(dtype=tf.int32, shape=[None], name="label_ids")
        self.training = tf.placeholder(dtype=tf.bool, name="training")

        self.build_model()
        self.init_saver()

    def build_model(self):
        bert_config = modeling.BertConfig.from_json_file(self.__bert_config_path)

        with tf.variable_scope("encoder"):
            outptu_layer = self.get_bert_encoder(bert_config, self.input_ids, self.input_masks, self.segment_ids,
                                               self.__dropout_rate)

        with tf.variable_scope("output"):
            self.logits = self.get_logits(outptu_layer)
            self.scores = tf.nn.softmax(self.logits, name="scores")

            self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")

        with tf.variable_scope("loss"):
            labels = tf.one_hot(self.label_ids, depth=self.__num_classes, dtype=tf.float32)

            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=labels)

            self.loss = tf.reduce_mean(losses, name="loss")

            if self.__use_r_drop:
                self.rdrop_loss = self.r_drop_loss(self.logits)
            else:
                self.rdrop_loss = 0.0

            self.total_loss = self.loss + self.rdrop_loss

        with tf.name_scope("train_op"):
            self.train_op = optimization.create_optimizer(self.total_loss, self.__learning_rate, self.__num_train_step,
                                                          self.__num_warmup_step, use_tpu=False)

    def get_logits(self, output_layer):
        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weight", [self.__num_classes, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02)
        )

        output_bias = tf.get_variable("output_bias", [self.__num_classes], initializer=tf.zeros_initializer())

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        return logits


    def get_bert_encoder(self, bert_config, input_ids, input_masks, segment_ids, dropout_rate=0.1):

        bert_config.attention_probs_dropout_prob = dropout_rate

        bert_config.hidden_dropout_prob = dropout_rate

        model = modeling.BertModel(config=bert_config,
                                   is_training=self.training,
                                   input_ids=input_ids,
                                   input_mask=input_masks,
                                   token_type_ids=segment_ids,
                                   use_one_hot_embeddings=False)

        output_layer = model.get_pooled_output()

        output_layer = tf.layers.dropout(output_layer, dropout_rate, training=self.training)

        return output_layer

    def lcml_loss(self, logtis, labels, scale=10, margin=0.35):
        norm_logits = tf.nn.l2_normalize(logtis, axis=1)

        new_logits = labels * (norm_logits - margin) + (1 - labels) * norm_logits

        new_logits *= scale

        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logtis=new_logits, labels=labels)

        return losses

    def r_drop_loss(self, logits, alpha=1):

        prob = tf.nn.softmax(logits, axis=-1)
        p, q = tf.split(prob, 2, axis=0)

        kl_loss =self.kl(p, q) + self.kl(q, p)

        kl_loss = tf.reduce_mean(kl_loss) / 2 * alpha

        return kl_loss

    def kl(self, y_true, y_pred, epsilon=1e-8):
        """
        计算 kl 散度
        :param y_true:
        :param y_pred:
        :param epsilon:
        :return:
        """

        log_y_true = tf.log(tf.clip_by_value(y_true, epsilon, 1. - epsilon))
        log_y_pred = tf.log(tf.clip_by_value(y_pred, epsilon, 1. - epsilon))

        kl_vals = tf.reduce_sum(y_true * log_y_true, axis=-1) - tf.reduce_sum(y_true * log_y_pred, axis=-1)

        return kl_vals

    def lebel_smoothing(self, label_ids):

        num_class = label_ids.get_shape().as_list()[-1]

        return ((1 - self.__smooth_rate) * label_ids) + (self.__smooth_rate / num_class)

    def init_saver(self):
        variables = tf.trainable_variables()

        variables = [variable for variable in variables if not re.search("adam", variable.name)]

        self.saver = tf.train.Saver(variables, max_to_keep=10)

    def train(self, sess, batch):
        """
        训练模型
        :param sess: tf的会话对象
        :param batch: batch数据
        :return: 损失和预测结果
        """

        if self.__use_r_drop:
            feed_dict = {
                self.input_ids: batch["input_ids"] + batch["input_ids"],
                self.input_masks: batch["input_masks"] + batch["input_masks"],
                self.segment_ids: batch["segment_ids"] + batch["segment_ids"],
                self.label_ids: batch["label_ids"] + batch["label_ids"],
                self.training: True
            }
        else:
            feed_dict = {
                self.input_ids: batch["input_ids"],
                self.input_masks: batch["input_masks"],
                self.segment_ids: batch["segment_ids"],
                self.label_ids: batch["label_ids"],
                self.training: True
            }

        # 训练模型
        _, loss, prediciotns = sess.run([self.train_op, self.total_loss, self.predictions], feed_dict=feed_dict)

        return loss, prediciotns

    def eval(self, sess, batch):
        """
        验证模型
        :param sess:
        :param batch:
        :return:
        """

        feed_dict = {
            self.input_ids: batch["input_ids"],
            self.input_masks: batch["input_masks"],
            self.segment_ids: batch["segment_ids"],
            self.label_ids: batch["label_ids"],
            self.training: True
        }

        loss, predictions = sess.run([self.loss, self.predictions], feed_dict=feed_dict)

        return loss, predictions








