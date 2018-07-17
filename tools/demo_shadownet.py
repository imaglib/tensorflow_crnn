#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Use shadow net to recognize the scene text
"""
import os
import sys
sys.path.append(os.getcwd())
import tensorflow as tf
import os.path as ops
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
try:
    from cv2 import cv2
except ImportError:
    pass

from crnn_model import crnn_model
from global_configuration import config
from local_utils import log_utils, data_utils

logger = log_utils.init_logger()


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='Where you store the image',
                        default='data/test_images/test_01.jpg')
    parser.add_argument('--weights_path', type=str, help='Where you store the weights',
                        default='model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999')

    return parser.parse_args()


def recognize(image_path, weights_path, is_vis=True):
    """

    :param image_path:
    :param weights_path:
    :param is_vis:
    :return:
    """

    #先把图片转为灰度图，然后缩放到config.cfg.TRAIN.IMAGE_WIDTH*32大小,添加batch和channel维度，然后每个像素转为float
    image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR),cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (config.cfg.TRAIN.IMAGE_WIDTH, 32)) 
    image = np.expand_dims(image, axis=0).astype(np.float32)
    image = np.expand_dims(image, axis=3).astype(np.float32)
    print(image.shape)

    inputdata = tf.placeholder(dtype=tf.float32, shape=[1, 32, config.cfg.TRAIN.IMAGE_WIDTH, 1], name='input')

    phase_tensor = tf.constant('test', tf.string)
    seq_length = tf.cast(config.cfg.TRAIN.SEQUENCE_LENGTH,tf.int32) 
    net = crnn_model.ShadowNet(phase=phase_tensor, hidden_nums=256, layers_nums=2, seq_length=seq_length,
                               num_classes=config.cfg.TRAIN.CLASSES_NUMS, rnn_cell_type='lstm')

    with tf.variable_scope('shadow'):
        net_out, tensor_dict = net.build_shadownet(inputdata=inputdata)

    # greedy_decoder根据当前序列预测下一个字符，并且取概率最高的作为结果，再此基础上再进行下一次预测。
    # 而beam_search_decoder每次会保存取k个概率最高的结果，以此为基础再进行预测，
    # 并将下一个字符出现的概率与当前k个出现的概率相乘，这样就可以减缓贪心造成的丢失好解的情况，
    # 当k=1的时候，二者就一样了
    decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=net_out, sequence_length=net.sequence_length*np.ones(1), merge_repeated=False)

    decoder = data_utils.TextFeatureIO()

    sess_config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4},log_device_placement=False,allow_soft_placement=True) 
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    # config tf saver
    saver = tf.train.Saver()

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        preds = sess.run(decodes, feed_dict={inputdata: image})

        preds = decoder.writer.sparse_tensor_to_str(preds[0])

        logger.info('Predict image {:s} label {:s}'.format(ops.split(image_path)[1], preds[0]))

        if is_vis:
            plt.figure('CRNN Model Demo')
            plt.imshow(cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, (2, 1, 0)])
            plt.show()

        sess.close()

    return


if __name__ == '__main__':
    # Inti args
    args = init_args()
    if not ops.exists(args.image_path):
        raise ValueError('{:s} doesn\'t exist'.format(args.image_path))

    # recognize the image
    recognize(image_path=args.image_path, weights_path=args.weights_path)
