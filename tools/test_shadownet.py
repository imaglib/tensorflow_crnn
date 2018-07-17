#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test shadow net script
"""
import os
import os.path as ops
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import numpy as np
import math
import sys
sys.path.append(os.getcwd())

from local_utils import data_utils
from crnn_model import crnn_model
from global_configuration import config


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Where you store the test tfrecords data')
    parser.add_argument('--weights_path', type=str, help='Where you store the shadow net weights')
    parser.add_argument('--is_recursive', type=bool, help='If need to recursively test the dataset')

    return parser.parse_args()


def test_shadownet(dataset_dir, weights_path, is_vis=False, is_recursive=True):
    """

    :param dataset_dir:
    :param weights_path:
    :param is_vis:
    :param is_recursive:
    :return:
    """
    # Initialize the record decoder
    decoder = data_utils.TextFeatureIO().reader
    images_t, labels_t, imagenames_t = decoder.read_features(dataset_dir, num_epochs=None, flag='Test')
    if not is_recursive:
        images_sh, labels_sh, imagenames_sh = tf.train.shuffle_batch(tensors=[images_t, labels_t, imagenames_t],
                                                                     batch_size=config.cfg.TRAIN.BATCH_SIZE, capacity=1000+config.cfg.TRAIN.BATCH_SIZE*2,
                                                                     min_after_dequeue=100, num_threads=2)
    else:
        images_sh, labels_sh, imagenames_sh = tf.train.batch(tensors=[images_t, labels_t, imagenames_t],
                                                             batch_size=config.cfg.TRAIN.BATCH_SIZE, capacity=1000 + config.cfg.TRAIN.BATCH_SIZE * 2, num_threads=2)

    images_sh = tf.cast(x=images_sh, dtype=tf.float32)

    # build shadownet
    phase_tensor = tf.constant('test', tf.string)
    seq_length = tf.cast(config.cfg.TRAIN.SEQUENCE_LENGTH,tf.int32)
    net = crnn_model.ShadowNet(phase=phase_tensor, hidden_nums=256, layers_nums=2, seq_length=seq_length,
                               num_classes=config.cfg.TRAIN.CLASSES_NUMS, rnn_cell_type='lstm')

    with tf.variable_scope('shadow'):
        net_out, tensor_dict = net.build_shadownet(inputdata=images_sh)

    decoded, _ = tf.nn.ctc_beam_search_decoder(net_out, net.sequence_length * np.ones(config.cfg.TRAIN.BATCH_SIZE), merge_repeated=False)

    sess_config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4},log_device_placement=False,allow_soft_placement=True) 
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    # config tf saver
    saver = tf.train.Saver()

    sess = tf.Session(config=sess_config)

    test_sample_count = 0
    # for record in tf.python_io.tf_record_iterator(ops.join(dataset_dir, 'test_feature_0_5000.tfrecords')):
    #     test_sample_count += 1
    # loops_nums = int(math.ceil(test_sample_count / config.cfg.TRAIN.BATCH_SIZE))
    loops_nums = 8

    with sess.as_default():

        # restore the model weights
        saver.restore(sess=sess, save_path=weights_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('Start predicting ......')
        if not is_recursive:
            predictions, images, labels, imagenames = sess.run([decoded, images_sh, labels_sh, imagenames_sh])
            imagenames = np.reshape(imagenames, newshape=imagenames.shape[0])
            imagenames = [tmp.decode('utf-8') for tmp in imagenames]
            preds_res = decoder.sparse_tensor_to_str(predictions[0])
            gt_res = decoder.sparse_tensor_to_str(labels)

            accuracy = []

            for index, gt_label in enumerate(gt_res):
                pred = preds_res[index]
                totol_count = len(gt_label)
                correct_count = 0
                try:
                    for i, tmp in enumerate(gt_label):
                        if tmp == pred[i]:
                            correct_count += 1
                except IndexError:
                    continue
                finally:
                    try:
                        accuracy.append(correct_count / totol_count)
                    except ZeroDivisionError:
                        if len(pred) == 0:
                            accuracy.append(1)
                        else:
                            accuracy.append(0)

            accuracy_val = np.mean(np.array(accuracy).astype(np.float32), axis=0)
            print('Test nums: {:d} mean accuracy is {:5f}'.format(len(accuracy), accuracy_val))

            for index, image in enumerate(images):
                print('Predict {:s} image with gt label: {:s} **** predict label: {:s}'.format(
                    imagenames[index], gt_res[index], preds_res[index]))
                if is_vis:
                    plt.imshow(image[:, :, (2, 1, 0)])
                    plt.show()
        else:
            accuracy = []
            for epoch in range(loops_nums):
                predictions, images, labels, imagenames = sess.run([decoded, images_sh, labels_sh, imagenames_sh])
                imagenames = np.reshape(imagenames, newshape=imagenames.shape[0])
                imagenames = [tmp.decode('utf-8') for tmp in imagenames]
                preds_res = decoder.sparse_tensor_to_str(predictions[0])
                gt_res = decoder.sparse_tensor_to_str(labels)

                for index, gt_label in enumerate(gt_res):
                    pred = preds_res[index]
                    totol_count = len(gt_label)
                    correct_count = 0
                    try:
                        for i, tmp in enumerate(gt_label):
                            if tmp == pred[i]:
                                correct_count += 1
                    except IndexError:
                        continue
                    finally:
                        try:
                            accuracy.append(correct_count / totol_count)
                        except ZeroDivisionError:
                            if len(pred) == 0:
                                accuracy.append(1)
                            else:
                                accuracy.append(0)

                for index, image in enumerate(images):
                    print('Predict {:s} image with gt label: {:s} **** predict label: {:s}'.format(
                        imagenames[index], gt_res[index], preds_res[index]))
                    # if is_vis:
                    #     plt.imshow(image[:, :, (2, 1, 0)])
                    #     plt.show()

            accuracy_val = np.mean(np.array(accuracy).astype(np.float32), axis=0)
            print('Test nums: {:d} accuracy is {:5f}'.format(len(accuracy), accuracy_val))

        coord.request_stop()
        coord.join(threads=threads)

    sess.close()
    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # args.dataset_dir     = 'data/tfrecords'
    # args.weights_path = 'model/shadownet/shadownet_2018-06-28-12-50-28.ckpt-8000'

    # test shadow net
    test_shadownet(args.dataset_dir, args.weights_path, args.is_recursive)
