#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 下午1:39
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : data_provider.py
# @IDE: PyCharm Community Edition
"""
Provide the training and testing data for shadow net
"""
import os.path as ops
import numpy as np
import copy

import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from data_provider import base_data_provider
from global_configuration import config


class TextDataset(base_data_provider.Dataset):
    """
        Implement a dataset class providing the image and it's corresponding text
    """
    def __init__(self, imageinfos,startIndex,endIndex, shuffle=None, normalization=None):
        """
        images, labels, imagenames
        :param images: image datasets [nums, H, W, C] 4D ndarray  实际图像的RGB数据
        :param labels: label dataset [nums, :] 2D ndarray  淮颂裔拎妻摇刊岗疡  午柳膘僚躇秃跳
        :param shuffle: if need shuffle the dataset, 'once_prior_train' represent shuffle only once before training
                        'every_epoch' represent shuffle the data every epoch
        :param imagenames:  Test\0\3.jpg  Test\0\6.jpg
        :param normalization: if need do normalization to the dataset,
                              'None': no any normalization
                              'divide_255': divide all pixels by 255
                              'divide_256': divide all pixels by 256


        test_images_org = [cv2.imread(tmp, cv2.IMREAD_COLOR)
                        for tmp in info[:, 0]]
        test_images = np.array([cv2.resize(tmp, (config.cfg.TRAIN.IMAGE_WIDTH, 32)) for tmp in test_images_org])

        test_labels = np.array([tmp for tmp in info[:, 1]]) 
        test_imagenames = np.array([tmp for tmp in info[:, 0]])

        train_images_org = [cv2.imread(tmp, cv2.IMREAD_COLOR)
                                for tmp in info[:, 0]]
        train_images = np.array([cv2.resize(tmp,(config.cfg.TRAIN.IMAGE_WIDTH,32)) for tmp in train_images_org])

        train_labels = np.array([tmp for tmp in info[:, 1]])
        train_imagenames = np.array([tmp for tmp in info[:, 0]])

        
        train_images_org = [cv2.imread(tmp, cv2.IMREAD_COLOR)
                            for tmp in info[:, 0]]
        train_images = np.array([cv2.resize(tmp, (config.cfg.TRAIN.IMAGE_WIDTH, 32)) for tmp in test_images_org])
        train_labels = np.array([tmp for tmp in info[:, 1]]) 
        train_imagenames = np.array([tmp for tmp in info[:, 0]])
         
        images=train_images[startIndex:endIndex], labels=train_labels[startIndex:endIndex],imagenames=train_imagenames[startIndex:endIndex]

        """
        super(TextDataset, self).__init__()

        self._imageinfos = copy.deepcopy(imageinfos[startIndex:endIndex])
        
        self.__normalization = normalization
        if self.__normalization not in [None, 'divide_255', 'divide_256']:
            raise ValueError('normalization parameter wrong')
        
        
        
        self._imageinfos = self.normalize_images(self._imageinfos, self.__normalization)
        self.__shuffle = shuffle
        if self.__shuffle not in [None, 'once_prior_train', 'every_epoch']:
            raise ValueError('shuffle parameter wrong') 

        if self.__shuffle == 'every_epoch' or 'once_prior_train':
            self._epoch_image_infos = self.shuffle_images_infos(self._imageinfos) 
 
        self.__batch_counter = 0
        return

    @property
    def num_examples(self):
        """

        :return:
        """
        return self._imageinfos.shape[0]
        # assert self.__images.shape[0] == self.__labels.shape[0]
        # return self.__labels.shape[0] 

    @property
    def images(self):
        """

        :return:
        """
        return self._epoch_images

    @property
    def labels(self):
        """

        :return:
        """
        return self._epoch_labels

    @property
    def imagenames(self):
        """

        :return:
        """
        return self._epoch_imagenames

    def next_batch(self, batch_size):
        """

        :param batch_size:
        :return:
        """
    
        start = self.__batch_counter * batch_size
        end = (self.__batch_counter + 1) * batch_size

        #print(" start = %d,end = %d,batch = %d" %(start,end,self.__batch_counter))

        self.__batch_counter += 1

        images_org = [cv2.cvtColor(cv2.imread(tmp, cv2.IMREAD_COLOR),cv2.COLOR_BGR2GRAY)
                        for tmp in self._epoch_image_infos[start : end, 0]]
        images = np.array([cv2.resize(tmp, (config.cfg.TRAIN.IMAGE_WIDTH, 32)) for tmp in images_org])
        labels = np.array([tmp for tmp in self._epoch_image_infos[start : end, 1]]) 
        imagenames = np.array([tmp for tmp in self._epoch_image_infos[start : end, 0]])  

        self.__images = images
        self.__labels = labels
        self.__imagenames = imagenames
        self._epoch_images = copy.deepcopy(self.__images)
        self._epoch_labels = copy.deepcopy(self.__labels)
        self._epoch_imagenames = copy.deepcopy(self.__imagenames)

        images_slice = self._epoch_images[:]
        labels_slice = self._epoch_labels[:]
        imagenames_slice = self._epoch_imagenames[:]
        # if overflow restart from the begining
        if images_slice.shape[0] != batch_size and images_slice.shape[0]!= self.num_examples:
            self.__start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice, labels_slice, imagenames_slice

    def __start_new_epoch(self):
        """

        :return:
        """
        self.__batch_counter = 0

        if self.__shuffle == 'every_epoch':
            self._epoch_image_infos = self.shuffle_images_infos(self._imageinfos)
        else:
            pass
        return


class TextDataProvider(object):
    """
        Implement the text data provider for training and testing the shadow net
    """
    def __init__(self, dataset_dir, annotation_name, validation_set=None, validation_split=None, shuffle=None,
                 normalization=None):
        """

        :param dataset_dir: str, where you save the dataset one class on folder
        :param annotation_name: annotation name
        :param validation_set:
        :param validation_split: `float` or None float: chunk of `train set` will be marked as `validation set`.
                                 None: if 'validation set' == True, `validation set` will be
                                 copy of `test set`
        :param shuffle: if need shuffle the dataset, 'once_prior_train' represent shuffle only once before training
                        'every_epoch' represent shuffle the data every epoch
        :param normalization: if need do normalization to the dataset,
                              'None': no any normalization
                              'divide_255': divide all pixels by 255
                              'divide_256': divide all pixels by 256
                              'by_chanels': substract mean of every chanel and divide each
                                            chanel data by it's standart deviation
        """
        self.__dataset_dir = dataset_dir
        self.__validation_split = validation_split
        self.__shuffle = shuffle
        self.__normalization = normalization
        self.__train_dataset_dir = ops.join(self.__dataset_dir, 'Train/')
        self.__test_dataset_dir = ops.join(self.__dataset_dir, 'Test/')

        assert ops.exists(dataset_dir)
        assert ops.exists(self.__train_dataset_dir)
        assert ops.exists(self.__test_dataset_dir)

        # add test dataset
        test_anno_path = ops.join(self.__test_dataset_dir, annotation_name)
        assert ops.exists(test_anno_path)

        with open(file=test_anno_path, mode='r',encoding='utf-8') as anno_file: 
            info = np.array([tmp.strip().split() for tmp in anno_file.readlines()]) 
            self.test = TextDataset(imageinfos=info,startIndex=0,endIndex=info.shape[0], shuffle=shuffle, normalization=normalization)             
        anno_file.close()

        # add train and validation dataset
        train_anno_path = ops.join(self.__train_dataset_dir, annotation_name)
        assert ops.exists(train_anno_path)

        with open(file=train_anno_path, mode='r',encoding='utf-8') as anno_file:
            info = np.array([tmp.strip().split() for tmp in anno_file.readlines()])

            if validation_set is not None and validation_split is not None:
                split_idx = int(info.shape[0] * (1 - validation_split))
                self.train = TextDataset(imageinfos=info,startIndex=0,endIndex=split_idx,
                                         shuffle=shuffle, normalization=normalization)
                self.validation = TextDataset(imageinfos=info, startIndex=split_idx,endIndex=info.shape[0]-split_idx,
                                              shuffle=shuffle, normalization=normalization)
            else:
                self.train = TextDataset(imageinfos=info, startIndex=0,endIndex=info.shape[0], shuffle=shuffle,
                                         normalization=normalization)


            if validation_set and not validation_split:
                self.validation = self.test
        anno_file.close()
        return

    def __str__(self):
        provider_info = 'Dataset_dir: {:s} contain training images: {:d} validation images: {:d} testing images: {:d}'.\
            format(self.__dataset_dir, self.train.num_examples, self.validation.num_examples, self.test.num_examples)
        return provider_info

    @property
    def dataset_dir(self):
        """

        :return:
        """
        return self.__dataset_dir

    @property
    def train_dataset_dir(self):
        """

        :return:
        """
        return self.__train_dataset_dir

    @property
    def test_dataset_dir(self):
        """

        :return:
        """
        return self.__test_dataset_dir
