# -*- encoding: utf-8 -*-
"""
@ Author: 钱朗
@ File: __init__.py
"""
import argparse
import collections
import datetime
import json
import os
import re
import time

import numpy as np
import tensorflow as tf

from bert import modeling
# from catePredict.textClassier.bert_softmax import data_helper

from model import BertClassifirer
from data_helper import TrainData

from metrics import mean, get_multi_metrics
from utils import get_logger


CUR_DIR = os.path.join(os.path.abspath(__file__))[0]

logger = get_logger("bert_softmax", "logs/log.txt")

def model_time():
    return str(datetime.datetime.now().strftime("%Y%m%d%H%M"))


class Trainer(object):

    def __init__(self, args):

        self.args = args

        with open(os.path.join(self.args.config_path), "r") as fr:
            self.config = json.load(fr)

        self.__bert_checkpoint_path = os.path.join(self.config['bert_model_path'], 'bert_model.ckpt')
        self.train_steps = self.config['train_steps']

        # 加载数据集
        self.data_obj = TrainData(self.config)

        self.t_in_ids, self.t_in_masks, self.t_seg_ids, self.t_lab_ids, lab_to_idx = self.data_obj.gen_data(
            self.config['train_data']
        )

        self.e_in_ids, self.e_in_masks, self.e_seg_ids, self.e_lab_ids, lab_to_idx = self.data_obj.gen_data(
            self.config['valid_data'], is_training=False
        )

        logger.info("trian_data_size:{}".format(len(self.t_in_ids)))
        logger.info("eval_data_size:{}".format(len(self.e_in_ids)))

        self.label_list = [value for key, value in lab_to_idx.items()]
        logger.info("label numbers:{}".format(len(self.label_list)))

        # 初始化模型
        self.model = BertClassifirer(config=self.config)

    def get_scope_vars(self, scope):
        variables = tf.trainable_variables()
        variables = [var for var in variables if scope in var.name]

        return variables

    def get_assignment_map_from_checkpoint(self, tvars, init_checkpoint, prefix_variable_scope=None):
        # assignment_map = {}
        name_to_variable = collections.OrderedDict()

        for var in tvars:
            name = var.name
            m = re.match("^(.*):\\d+$", name)

            if m is not None:
                name = m.group(1)

            name_to_variable[name] = var

        init_vars = tf.train.list_variables(init_checkpoint)

        assignment_map = collections.OrderedDict()

        for x in init_vars:
            (name, var) = (x[0], x[1])

            if prefix_variable_scope:
                new_name = prefix_variable_scope + "/" + name
            else:
                new_name = name
            if new_name not in name_to_variable:
                continue

            assignment_map[name] = new_name
        return assignment_map


    def train(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)

        with tf.Session(config=sess_config) as sess:
            logger.info("init bert model params")
            # tvars = tf.trainable_variables()
            tvars = self.get_scope_vars('encoder')

            assignment_map = self.get_assignment_map_from_checkpoint(
                tvars, self.__bert_checkpoint_path, 'encoder'
            )

            # tf.train.init_from_checkpoint(self.__bert_checkpoint_path, assignment_map)

            # (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
            #     tvars, self.__bert_checkpoint_path
            # )
            # logger.info("init bert model params")

            tf.train.init_from_checkpoint(self.__bert_checkpoint_path, assignment_map)
            logger.info("init bert model params done")

            sess.run(tf.variables_initializer(tf.global_variables()))

            current_step = 0
            start_time = time.time()

            for epoch in range(self.config['epochs']):
                logger.info("----- epoch{}/{} -----".format(epoch + 1, self.config["epochs"]))

                for batch in self.data_obj.next_batch(self.t_in_ids, self.t_in_masks, self.t_seg_ids, self.t_lab_ids):

                    loss, predictions = self.model.train(sess, batch)

                    if current_step % self.config['log_every'] == 0:

                        true_y = batch['label_ids']

                        sample_num = len(true_y)

                        predictions = predictions.tolist()
                        pred_y = predictions[:sample_num]

                        acc, recall, prec, f_beta = get_multi_metrics(
                            pred_y=pred_y, true_y=true_y, labels=self.label_list
                        )

                        logger.info("trian: step:{}, loss: {}, acc:{}, recall:{}, pred:{}, f_beta:{}".format(
                            current_step, loss, acc, recall, prec, f_beta
                        ))

                    current_step += 1

                    if self.data_obj and current_step % self.config['eval_every'] == 0:

                        eval_losses = []
                        eval_preds = []
                        eval_labels = []

                        for eval_batch in self.data_obj.next_batch(self.e_in_ids, self.e_in_masks, self.e_seg_ids, self.e_lab_ids):

                            eval_losse, eval_preditions = self.model.eval(sess, eval_batch)

                            true_y = eval_batch['label_ids']
                            sample_num = len(true_y)

                            eval_preditions = eval_preditions.tolist()
                            pred_y = eval_preditions[:sample_num]

                            eval_losses.append(eval_losse)
                            eval_preds.append(pred_y)
                            eval_labels.append(true_y)

                        acc, recall, prec, f_beta = get_multi_metrics(
                            pred_y=eval_preds, true_y=eval_labels, labels=self.label_list
                        )

                        logger.info('\n')
                        # logger.info("eval_loss{}, acc:{}, macro_f1:{}, micro_f1:{}".format(mean(eval_losses, acc, macro_f1, mirco_f1)))
                        logger.info("trian: step:{}, loss: {}, acc:{}, recall:{}, pred:{}, f_beta:{}".format(
                            current_step, loss, acc, recall, prec, f_beta
                        ))
                        logger.info('\n')

                    if current_step % self.config['checkpoint_every'] == 0:
                        if self.config['ckpt_model_path']:
                            save_path = self.config['ckpt_model_path']

                            if not os.path.exists(save_path):
                                os.makedirs(save_path)

                            model_save_path = os.path.join(save_path, self.config['model_name'])
                            self.model.saver.save(sess, model_save_path, global_step=current_step)

                    if current_step > self.train_steps:
                        break

                if current_step > self.train_steps:
                    break

            end_time = time.time()

            print('total train time', end_time - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',default='config.json', help='config path of model')

    args, unparsed = parser.parse_known_args()

    trainer = Trainer(args)
    trainer.train()
















         