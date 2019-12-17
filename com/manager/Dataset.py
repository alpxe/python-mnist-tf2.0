from com.core.base.Singleton import Singleton
from com.manager.Download import Download

import gzip
import struct
import os
import numpy as np
import sys
import tensorflow as tf


class Dataset(Singleton):
    TFrecords = "assets/tfrecords"
    TRAIN_FILE = "train.tfrecord"

    def __single__(self):
        """ override __single__"""

        self.image_path = os.path.join(Download.dest_directory, Download.TRAIN_IMAGES)
        self.label_path = os.path.join(Download.dest_directory, Download.TRAIN_LABELS)
        self.train_path = os.path.join(self.TFrecords, self.TRAIN_FILE)

        if not os.path.exists(self.TFrecords):
            os.makedirs(self.TFrecords)  # 生成文件夹

    def create_train(self):
        """
        创建训练集
        """

        if os.path.exists(self.image_path):
            img_buf = self.__open_path(self.image_path)
        if os.path.exists(self.label_path):
            label_buf = self.__open_path(self.label_path)

        # 获取字节流中，前4个I。<>,<样本数量>,<样本纬度>,<样本纬度>
        magic, items, row, col = struct.unpack_from(">IIII", img_buf, 0)

        img_hd = struct.calcsize(">IIII")
        label_hd = struct.calcsize(">II")

        # 生成 .tfrecords
        with tf.io.TFRecordWriter(self.train_path) as writer:
            for i in range(items):  # 60000个样本
                label, = struct.unpack_from(">B", label_buf, label_hd + i * 1)  # ->tuple >(B,)
                img = struct.unpack_from(">{0}B".format(row * col), img_buf, img_hd + i * row * col)

                fs = tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    "image": tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(img)))
                })
                example = tf.train.Example(features=fs)
                writer.write(example.SerializeToString())  # SerializeToString 写入

                # 打印进度
                sys.stdout.write("\rWrite progress： {0:.2f}%".format((i + 1) / items * 100))
                sys.stdout.flush()

    def read(self):
        """
        获取训练集
        """
        if not os.path.exists(self.train_path):  # 如果文件不存在
            self.create_train()  # 创建训练集

        dataset = tf.data.TFRecordDataset(self.train_path)

        dataset = dataset.repeat()
        dataset = dataset.map(self.__format)
        dataset = dataset.batch(10)

        return dataset

    @staticmethod
    def __format(record):
        return tf.io.parse_single_example(record, features={
            "label": tf.io.FixedLenFeature([1], dtype=tf.int64),
            "image": tf.io.FixedLenFeature([28 * 28], dtype=tf.int64)
        })

    @staticmethod
    def __open_path(path):
        with open(path, 'rb') as f:  # 打开文件
            with gzip.GzipFile(fileobj=f) as bytestream:  # 解压缩
                return bytestream.read()
