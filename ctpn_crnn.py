from demo import CTPN
import numpy as np
import sys,os
import glob
import mahotas
import shutil
import matplotlib.pyplot as plt
from PIL import Image 
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

from math import atan2,degrees,fabs,sin,radians,cos

sys.path.append(os.getcwd())


class CRNN(object):
     #load the model
    def __init__(self):
        print("CRNN init") 

    def call_crnn_rpc(self,image_data):
        
        return 

      
class CTPN_CRNN(object):
      #load the model
    def __init__(self): 
        print("CTPN CRNN Init")

    # 文本检测
    def text_detection(self,im,image_name):
        ctpn = CTPN()
        img,text_recs = ctpn.get_text_box(im,image_name)
        return img, text_recs
    
    def dumpRotateImage(self,img, degree, pt1, pt2, pt3, pt4):
        height, width = img.shape[:2]
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
        matRotation[0, 2] += (widthNew - width) / 2
        matRotation[1, 2] += (heightNew - height) / 2
        imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
        pt1 = list(pt1)
        pt3 = list(pt3)

        [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
        [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
        imgOut = imgRotation[int(pt1[1]):int(pt3[1]), int(pt1[0]):int(pt3[0])]
        height, width = imgOut.shape[:2]
        return imgOut
        
    def crnnRec(self, im, text_recs):
        index = 0 
        images = []
        for rec in text_recs:
            pt1 = (rec[0], rec[1])
            pt2 = (rec[2], rec[3])
            pt3 = (rec[6], rec[7])
            pt4 = (rec[4], rec[5])
            partImg = self.dumpRotateImage(im, degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])), pt1, pt2, pt3, pt4)
            if(partImg.shape[0]==0 or partImg.shape[1]==0 or partImg.shape[2]==0):
                continue


            #mahotas.imsave('data/tmp/%s.jpg'%index, partImg) 
            
            # image = Image.open('data/test.jpg').convert('L') 
            #先转灰度再去做识别
            image = Image.fromarray(partImg).convert('L')
            #image.save('data/tmp/gray_%s.jpg'%index)
            height,width,channel=partImg.shape[:3]
            print(height,width,channel)
            print(image.size)

            #调整为width*32大小的图，CRNN都是基于不定长等高32的样本训练
            scale = image.size[1] * 1.0 / 32
            w = image.size[0] / scale
            w = int(w) 

            rcnnImgSize =  w,32
            image = image.resize(rcnnImgSize, Image.ANTIALIAS)
            #得到了合适大小的图片，交给CRNN去做识别 
            im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
            #im_arr = im_arr.reshape((image.size[1], image.size[0], 1))   
             
            images.append(im_arr)
            index += 1


        crnn_model = CRNN()   
        for i in range(len(images)):
            preds.append(crnn_model.call_crnn_rpc(images[i]))

        return preds 

    # 文本识别 
    def text_recognition(self,img, text_recs): 
        preds = self.crnnRec(im=img, text_recs=text_recs)
        return preds

    def do(self,img_name,is_show=True):
        print("---------------------------------------------------------------")
        print("start to recognize : %s"%img_name)
        # 读取图片
        im = cv2.imread(img_name)
        # 利用CTPN检测文本块
        img, text_recs = self.text_detection(im,img_name)
        # 使用CRNN识别文本
        preds = self.text_recognition(img, text_recs)
        # 输出识别结果
        for i in range(len(preds)):
            print("%s" % (preds[i]))
        print("---------------------------------------------------------------")

        # Matplotlib pyplot.imshow(): M x N x 3 image, where last dimension is RGB.
        # OpenCV cv2.imshow(): M x N x 3 image, where last dimension is BGR
        # plt.imshow(cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, (2, 1, 0)])
        if(is_show):
            srcBGR = img.copy()
            destRGB = cv2.cvtColor(srcBGR,cv2.COLOR_BGR2RGB)
            plt.imshow(destRGB)
            plt.show()


def ctpn_crnn_do(ctpn_crnn,im_name): 
    ctpn_crnn.do(im_name,False)

if __name__ == '__main__':
    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results/") 
 
    ctpn_crnn = CTPN_CRNN() 
    img_names = glob.glob(os.path.join("./data/demo",'*.jpg'))+\
                    glob.glob(os.path.join("./data/demo",'*.png'))+\
                    glob.glob(os.path.join("./data/demo",'*.bmp'))
    for im_name in img_names:
        ctpn_crnn_do(ctpn_crnn,im_name)
    
