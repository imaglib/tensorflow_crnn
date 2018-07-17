#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午3:25
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : config.py
# @IDE: PyCharm Community Edition
"""
Set some global configuration
"""
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# Train options
__C.TRAIN = edict()

# Set the shadownet training epochs
__C.TRAIN.EPOCHS = 3000000
# Set the display step
__C.TRAIN.DISPLAY_STEP = 1
# Set the test display step during training process
__C.TRAIN.TEST_DISPLAY_STEP = 100
# Set the momentum parameter of the optimizer
__C.TRAIN.MOMENTUM = 0.9
# Set the GPU resource used during training process
__C.TRAIN.GPU_MEMORY_FRACTION = 0.90
# Set the GPU allow growth parameter during tensorflow training process
__C.TRAIN.TF_ALLOW_GROWTH = True
# Set the shadownet training batch size
__C.TRAIN.BATCH_SIZE = 256
# Set the shadownet validation batch size
__C.TRAIN.VAL_BATCH_SIZE = 256
# Set the initial learning rate
__C.TRAIN.LEARNING_RATE = 0.09
# Set the learning rate decay steps
__C.TRAIN.LR_DECAY_STEPS = 60000
# Set the learning rate decay rate
__C.TRAIN.LR_DECAY_RATE = 0.3
# Set the class numbers
__C.TRAIN.CLASSES_NUMS = 3817+1
# Set the validation step
__C.TRAIN.VAL_STEP = 100
# Set the accuracy epochs
__C.TRAIN.ACC_STEP = 10
# Set the train image width
__C.TRAIN.IMAGE_WIDTH = 256
# Set the train sequence length
__C.TRAIN.SEQUENCE_LENGTH = 40

# Test options
__C.TEST = edict()

# Set the GPU resource used during testing process
__C.TEST.GPU_MEMORY_FRACTION = 0.5
# Set the GPU allow growth parameter during tensorflow testing process
__C.TEST.TF_ALLOW_GROWTH = False
# Set the test batch size
__C.TEST.BATCH_SIZE = 32
