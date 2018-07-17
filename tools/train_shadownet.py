#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午1:39
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : train_shadownet.py
# @IDE: PyCharm Community Edition
"""
Train shadow net script
"""
import argparse
import os
import os.path as ops
import sys
import time
import datetime
from tensorflow.python.client import timeline
import numpy as np
import tensorflow as tf
import pprint
import math 

sys.path.append(os.getcwd())

from crnn_model import crnn_model
from local_utils import data_utils, log_utils, tensorboard_vis_summary
from global_configuration import config


logger = log_utils.init_logger()


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Where you store the dataset')
    parser.add_argument('--train_num', type=int, help='how many samples in the dataset')
    parser.add_argument('--weights_path', type=str, help='Where you store the pretrained weights')

    return parser.parse_args()


def train_shadownet(dataset_dir,train_num, weights_path=None):
    """

    :param dataset_dir:
    :param weights_path:
    :return:
    """
    # input_tensor = tf.placeholder(dtype=tf.float32, shape=[config.cfg.TRAIN.BATCH_SIZE, 32, config.cfg.TRAIN.IMAGE_WIDTH, 3],
    #                               name='input_tensor')

    # decode the tf records to get the training data
    decoder = data_utils.TextFeatureIO().reader

    #读取tfrecords中的图像数据以及对应原文字标签,图片路径
    images, labels, imagenames = decoder.read_features(dataset_dir, num_epochs=None,
                                                       flag='Train')
    # images_val, labels_val, imagenames_val = decoder.read_features(dataset_dir, num_epochs=None,
    #                                                  flag='Validation')
    
    #将数据按指定批次大小进行随机打散
    inputdata, input_labels, input_imagenames = tf.train.shuffle_batch(
        tensors=[images, labels, imagenames], batch_size=config.cfg.TRAIN.BATCH_SIZE,
        capacity=1000 + 2 * config.cfg.TRAIN.BATCH_SIZE, min_after_dequeue=100, num_threads=1)

    # inputdata_val, input_labels_val, input_imagenames_val = tf.train.shuffle_batch(
    #     tensors=[images_val, labels_val, imagenames_val], batch_size=config.TRAIN.BATCH_SIZE,
    #     capacity=1000 + 2 * config.TRAIN.BATCH_SIZE, min_after_dequeue=100, num_threads=1)

    inputdata = tf.cast(x=inputdata, dtype=tf.float32)
    phase_tensor = tf.placeholder(dtype=tf.string, shape=None, name='phase')
    accuracy_tensor = tf.placeholder(dtype=tf.float32, shape=None, name='accuracy_tensor')

    #这个公式要根据实际卷积和池化的参数来决定
    seq_length = tf.cast(config.cfg.TRAIN.SEQUENCE_LENGTH,tf.int32)
    # initialize the net model
    shadownet = crnn_model.ShadowNet(phase=phase_tensor, hidden_nums=256, layers_nums=2, seq_length=seq_length,
                                     num_classes=config.cfg.TRAIN.CLASSES_NUMS, rnn_cell_type='lstm')

    with tf.variable_scope('shadow', reuse=False):
        net_out, tensor_dict = shadownet.build_shadownet(inputdata=inputdata)

    #求ctc_loss损失平均值
    cost = tf.reduce_mean(tf.nn.ctc_loss(labels=input_labels, inputs=net_out,
                                         sequence_length=shadownet.sequence_length*np.ones(config.cfg.TRAIN.BATCH_SIZE)))

    #前面说的划分块之后找每块的类属概率分布，ctc_beam_search_decoder方法,是每次找最大的K个概率分布
    #还有一种贪心策略是只找概率最大那个，也就是K=1的情况ctc_greedy_decoder
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(net_out,
                                                      shadownet.sequence_length*np.ones(config.cfg.TRAIN.BATCH_SIZE),
                                                      merge_repeated=False)
    #计算序列之间的编辑距离的平均值
    sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), input_labels))

    #每多少步就更新梯度
    global_step = tf.Variable(0, name='global_step', trainable=False)

    starter_learning_rate = config.cfg.TRAIN.LEARNING_RATE
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               config.cfg.TRAIN.LR_DECAY_STEPS, config.cfg.TRAIN.LR_DECAY_RATE,
                                               staircase=True)

    #需要显示调用训练时得出的参数
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #必须算完update_ops后再去更新梯度
    with tf.control_dependencies(update_ops):
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=cost, global_step=global_step)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=cost, global_step=global_step)
        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(
        #     loss=cost, global_step=global_step)

    # Set tf summary
    tboard_save_path = 'tboard/shadownet'
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)

    visualizor = tensorboard_vis_summary.CNNVisualizer()

    # training过程summary
    train_cost_scalar = tf.summary.scalar(name='train_cost', tensor=cost)
    train_accuracy_scalar = tf.summary.scalar(name='train_accuray', tensor=accuracy_tensor)
    train_seq_scalar = tf.summary.scalar(name='train_seq_dist', tensor=sequence_dist)
    train_conv1_image = visualizor.merge_conv_image(feature_map=tensor_dict['conv1'],
                                                    scope='conv1_image')
    train_conv2_image = visualizor.merge_conv_image(feature_map=tensor_dict['conv2'],
                                                    scope='conv2_image')
    train_conv3_image = visualizor.merge_conv_image(feature_map=tensor_dict['conv3'],
                                                    scope='conv3_image')
    lr_scalar = tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    
    weights_tensor_dict = dict()
    for vv in tf.trainable_variables():
        if 'conv' in vv.name:
            weights_tensor_dict[vv.name[:-2]] = vv
    train_weights_hist_dict = visualizor.merge_weights_hist(
        weights_tensor_dict=weights_tensor_dict, scope='weights_histogram', is_merge=False)

    train_summary_merge_list = [train_cost_scalar, train_accuracy_scalar, train_seq_scalar, lr_scalar,
                                train_conv1_image, train_conv2_image, train_conv3_image]
    for _, weights_hist in train_weights_hist_dict.items():
        train_summary_merge_list.append(weights_hist)
    train_summary_op_merge = tf.summary.merge(inputs=train_summary_merge_list)

    # validation过程summary
    # val_cost_scalar = tf.summary.scalar(name='val_cost', tensor=cost)
    # val_seq_scalar = tf.summary.scalar(name='val_seq_dist', tensor=sequence_dist)
    # val_accuracy_scalar = tf.summary.scalar(name='val_accuracy', tensor=accuracy_tensor)

    # test_summary_op_merge = tf.summary.merge(inputs=[val_cost_scalar, val_accuracy_scalar,
    #                                                  val_seq_scalar])

    # Set saver configuration
    restore_variable_list = [tmp.name for tmp in tf.trainable_variables()]
    saver = tf.train.Saver()
    model_save_dir = 'model/shadownet'
    #export_path = 'model/service'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)
 
    # Set sess configuration ,device_count  
    sess_config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4},log_device_placement=False,allow_soft_placement=True) 
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs_iteration = config.cfg.TRAIN.EPOCHS*config.cfg.TRAIN.BATCH_SIZE

    print('Global configuration is as follows:')
    pprint.pprint(config.cfg)

    #builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    with sess.as_default():

        if weights_path is None:
            logger.info('Training from scratch')
            init = tf.global_variables_initializer()

            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            # sess.run(init,options=run_options, run_metadata=run_metadata)
            #  # Create the Timeline object, and write it to a json
            # tl = timeline.Timeline(run_metadata.step_stats)
            # ctf = tl.generate_chrome_trace_format()
            # with open('d:\\timeline.json', 'w') as f:
            #     f.write(ctf)
            sess.run(init)
        else:
            # logger.info('Restore model from last crnn check point{:s}'.format(weights_path))
            # init = tf.global_variables_initializer()
            # sess.run(init)
            # restore_saver = tf.train.Saver(var_list=restore_variable_list)
            # restore_saver.restore(sess=sess, save_path=weights_path)
            logger.info('Restore model from last crnn check point{:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        total_batch = int(math.ceil(train_num/config.cfg.TRAIN.BATCH_SIZE))
        accuracy_cacul_iterations = config.cfg.TRAIN.ACC_STEP * total_batch

        epoch = 0
        

        for iteration in range(train_epochs_iteration):
            epoch = int(iteration/total_batch)

            if iteration % accuracy_cacul_iterations == 0:
                _, c, seq_distance, preds, gt_labels = sess.run(
                    [optimizer, cost, sequence_dist, decoded, input_labels],
                    feed_dict={phase_tensor: 'train'})

                 
                # calculate the precision
                preds = decoder.sparse_tensor_to_str(preds[0])
                gt_labels = decoder.sparse_tensor_to_str(gt_labels)

                accuracy = []

                for index, gt_label in enumerate(gt_labels):
                    pred = preds[index]
                    totol_count = len(gt_label)
                    correct_count = 0
                    print('index {:d}  gt_label {:s} --- preds {:s}'.format(index , gt_label, pred))
                    # print(" gt_label %s --- preds %s" % (gt_label,pred))
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
                accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)

                

                train_summary = sess.run(train_summary_op_merge,
                                        feed_dict={accuracy_tensor: accuracy,
                                                    phase_tensor: 'train'})

                summary_writer.add_summary(summary=train_summary, global_step=epoch)
 
                logger.info('Epoch: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
                        epoch + 1, c, seq_distance, accuracy)) 
                

            else:
                _, c = sess.run(
                    [optimizer, cost],
                    feed_dict={phase_tensor: 'train'})
 

                if iteration % config.cfg.TRAIN.DISPLAY_STEP == 0: 
                    logger.info('Epoch: {:d} iteration:{:d}/{:d}/{:d} cost= {:9f} '.format(
                        epoch + 1,total_batch,iteration,accuracy_cacul_iterations, c))

            
            if iteration % 2000 == 0 :
                saver.save(sess=sess, save_path=model_save_path, global_step=iteration) 
                print('save model {:d}'.format(iteration))            
             

        coord.request_stop()
        coord.join(threads=threads)

     
#   # Build the signature_def_map.
#     classification_inputs = tf.saved_model.utils.build_tensor_info(
#         serialized_tf_example)
#     classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
#         prediction_classes)
#     classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

#     classification_signature = (
#         tf.saved_model.signature_def_utils.build_signature_def(
#             inputs={
#                 tf.saved_model.signature_constants.CLASSIFY_INPUTS:
#                     classification_inputs
#             },
#             outputs={
#                 tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
#                     classification_outputs_classes,
#                 tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
#                     classification_outputs_scores
#             },
#             method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

#     tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
#     tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

#     prediction_signature = (
#         tf.saved_model.signature_def_utils.build_signature_def(
#             inputs={'images': tensor_info_x},
#             outputs={'scores': tensor_info_y},
#             method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

#     legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
#     builder.add_meta_graph_and_variables(
#         sess, [tf.saved_model.tag_constants.SERVING],
#         signature_def_map={
#             'predict_images':
#                 prediction_signature,
#             tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
#                 classification_signature,
#         },
#         legacy_init_op=legacy_init_op)

#     builder.save()   

    sess.close()

    return


if __name__ == '__main__':
    # decoder = data_utils.TextFeatureIO().reader
    # b = decoder.int_to_char(84) 
    #* a = decoder.int_to_char(84)
    # b = decoder.int_to_char('84')
    # c = decoder.int_to_char(1)
    # d = decoder.int_to_char('1')

    # init args
    args = init_args()
    #args.dataset_dir ='data/tfrecords/' 
    #args.train_num = 950000
    #args.weights_path ='model/shadownet/shadownet_2018-05-30-11-52-19.ckpt-28000'  

    if not ops.exists(args.dataset_dir):
        raise ValueError('{:s} doesn\'t exist'.format(args.dataset_dir))

    print('class number {:d}'.format(config.cfg.TRAIN.CLASSES_NUMS))
    train_shadownet(args.dataset_dir, args.train_num, args.weights_path)

    # if args.weights_path is not None and 'two_stage' in args.weights_path:
    #     train_shadownet(args.dataset_dir, args.weights_path, restore_from_cnn_subnet_work=False)
    # elif args.weights_path is not None and 'cnnsub' in args.weights_path:
    #     train_shadownet(args.dataset_dir, args.weights_path, restore_from_cnn_subnet_work=True)
    # else:
    #     train_shadownet(args.dataset_dir)
    print('Done')
