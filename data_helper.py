# -*- encoding: utf-8 -*-
"""
@ Author: 钱朗
@ File: __init__.py
"""

import os
import random

import tensorflow as tf
from bert import tokenization

# import pandas as pd
# from sklearn.utils import shuffle


class TrainData(object):
    def __init__(self, config):
        self.__vocab_path = os.path.join(config['bert_model_path'], "vocab.txt")
        self.__output_path = config['output_path']

        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)

        self._sequence_length = config['sequence_length']

        self._batch_size = config['batch_size']

    @staticmethod
    def read_data(file_path):
        """
        读取数据
        :param file_path:
        :return:
        """

        inputs = []
        labels = []

        with tf.gfile.GFile(file_path, 'r') as fr:
            for line in fr.readlines():
                item = line.strip().split('\t')

                if item[1].strip() == '':
                    continue

                inputs.append(item[1])
                labels.append(item[0])

        return inputs, labels

    def trans_to_index(self, inputs):
        """
        将输入转化为索引
        :param inputs:输入
        :return:
        """

        print("vocab_path:", self.__vocab_path)

        tokenizer = tokenization.FullTokenizer(vocab_file=self.__vocab_path, do_lower_case=True)

        input_ids = []
        input_masks = []
        segment_ids = []

        for text in inputs:
            text = tokenization.convert_to_unicode(text)
            tokens = tokenizer.tokenize(text)
            tokens = ['[CLS]'] + tokens + ['[SEP]']

            input_id = tokenizer.convert_tokens_to_ids(tokens)

            input_ids.append(input_id)
            input_masks.append([1] * len(input_id))
            segment_ids.append([0] * len(input_id))

        return input_ids, input_masks, segment_ids

    @staticmethod
    def trans_label_to_index(labels, label_to_index):
        """
        将标签也转换成数字表示
        :param labels:标签
        :param label_to_index: 标签-索引映射表
        :return:
        """
        label_idx = [label_to_index[label] for label in labels]
        return label_idx

    def padding(self, input_ids, input_masks, segment_ids):
        """
        对序列进行补全
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :return:
        """

        pad_input_ids, pad_input_masks, pad_segment_ids = [], [], []

        for input_id, input_mask, segment_id in zip(input_ids, input_masks, segment_ids):
            if len(input_id) < self._sequence_length:
                pad_input_ids.append(input_id + [0] * (self._sequence_length - len(input_id)))
                pad_input_masks.append(input_mask + [0] * (self._sequence_length - len(input_mask)))
                pad_segment_ids.append(segment_id + [0] * (self._sequence_length - len(segment_id)))
            else:
                pad_input_ids.append(input_id[:self._sequence_length])
                pad_input_masks.append(input_mask[:self._sequence_length])
                pad_segment_ids.append(segment_id[:self._sequence_length])

        return pad_input_ids, pad_input_masks, pad_segment_ids

    def gen_data(self, file_path, is_training=True):
        """
        生成数据
        :param file_path:
        :param is_training:
        :return:
        """

        # 读取原始数据
        inputs, labels = self.read_data(file_path)

        if is_training:
            uni_label = list(set(labels))
            uni_label = sorted(uni_label)

            label_to_index = dict(zip(uni_label, list(range(len(uni_label)))))

            with tf.gfile.GFile(os.path.join(self.__output_path, 'label_to_index.txt'), 'w') as fw:
                label_save = [key + '\t' + str(value) for key, value in label_to_index.items()]
                fw.write("\n".join(label_save))

        else:
            label_to_index = {}

            with tf.gfile.GFile(os.path.join(self.__output_path, 'label_to_index.txt'), 'r') as fr:
                for line in fr:
                    item = line.strip().split('\t')
                    label_to_index[item[0]] = int(item[1])

        # 输入转索引
        inputs_ids, inputs_masks, segment_ids = self.trans_to_index(inputs)

        inputs_ids, inputs_masks, segment_ids = self.padding(inputs_ids, inputs_masks, segment_ids)

        # 标签转索引
        labels_ids = self.trans_label_to_index(labels, label_to_index)
        print("label index transform finished")

        for i in range(5):
            print("line {}: ************************************".format(i))
            print('input_id:', inputs_ids[i])
            print("input_id_len", len(inputs_ids[i]))
            print('inputs_mask:', inputs_masks[i])
            print("input_id_len", len(inputs_masks[i]))
            print('segment_id:', segment_ids[i])
            print("input_id_len", len(segment_ids[i]))
            print('label_id:', labels_ids[i])

        # print('labels_ids', labels_ids)

        return inputs_ids, inputs_masks, segment_ids, labels_ids, label_to_index

    def next_batch(self, input_ids, input_masks, segment_ids, label_ids, is_training=True):
        """
        生成 batch 数据
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :param label_ids:
        :param is_training:
        :return:
        """
        z = list(zip(input_ids, input_masks, segment_ids, label_ids))

        random.shuffle(z)

        input_ids, input_masks, segment_ids, label_ids = zip(*z)

        num_batches = len(input_ids) // self._batch_size

        if not is_training:
            num_batches += 1

        for i in range(num_batches):
            start = i * self._batch_size
            end = start + self._batch_size

            batch_input_ids = input_ids[start:end]
            batch_input_masks = input_masks[start:end]
            batch_segment_ids = segment_ids[start:end]
            batch_label_ids = label_ids[start:end]

            yield dict(input_ids=batch_input_ids, input_masks=batch_input_masks,
                       segment_ids=batch_segment_ids, label_ids=batch_label_ids)
