
import os.path as ops
import numpy as np

import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

if __name__ == '__main__':
 with open(file='Test/labels.txt', mode='r',encoding='utf-8') as anno_file: 
            info = np.array([tmp.strip().split() for tmp in anno_file.readlines()]) 

            test_images_org = [cv2.imread(tmp, cv2.IMREAD_COLOR)
                               for tmp in info[:, 0]]
            test_images = np.array([cv2.resize(tmp, (config.cfg.TRAIN.IMAGE_WIDTH, 32)) for tmp in test_images_org])

            test_labels = np.array([tmp for tmp in info[:, 1]]) 
            test_imagenames = np.array([tmp for tmp in info[:, 0]])
            print('ok')