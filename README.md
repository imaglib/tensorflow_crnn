# CRNN_Tensorflow
Use tensorflow to implement a Deep Neural Network for scene text recognition mainly based on the paper "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition".You can refer to their paper for details http://arxiv.org/abs/1507.05717. Thanks for the author [Baoguang Shi](https://github.com/bgshih).  
This model consists of a CNN stage, RNN stage and CTC loss for scene text recognition task.

## Installation
This software has only been tested on ubuntu 16.04(x64), python3.5, cuda-8.0, cudnn-6.0 with a GTX-1070 GPU. To install this software you need tensorflow 1.3.0 and other version of tensorflow has not been tested but I think it will be to work properly in tensorflow above version 1.0. Other required package you may install them by

```
pip3 install -r requirements.txt
```

## Test model
In this repo I uploaded a model trained on a subset of the [Synth 90k](http://www.robots.ox.ac.uk/~vgg/data/text/). During data preparation process the dataset is converted into a tensorflow records which you can find in the data folder.
You can test the trained model on the converted dataset by

```
python tools/test_shadownet.py --dataset_dir data/ --weights_path model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999
```
`Expected output is`  
![Test output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_output.png)
If you want to test a single image you can do it by
```
python tools/demo_shadownet.py --image_path data/test_images/test_01.jpg --weights_path model/shadownet/shadownet_2017-09-29-19-16-33.ckpt-39999
``` 

## Train your own model
#### Data Preparation
Firstly you need to store all your image data in a root folder then you need to supply a txt file named sample.txt to specify the relative path to the image data dir and it's corresponding text label. For example

```
path/1/2/373_coley_14845.jpg coley
path/17/5/176_Nevadans_51437.jpg nevadans
```

Secondly you are supposed to convert your dataset into tensorflow records which can be done by
```
python tools/write_text_features --dataset_dir path/to/your/dataset --save_dir path/to/tfrecords_dir
```
All your training image will be scaled into (32, 100, 3) the dataset will be divided into train, test, validation set and you can change the parameter to control the ratio of them.

#### Train model
The whole training epoches are 40000 in my experiment. I trained the model with a batch size 32, initialized learning rate is 0.1 and decrease by multiply 0.1 every 10000 epochs. About training parameters you can check the global_configuration/config.py for details. To train your own model by

```
python tools/train_shadownet.py --dataset_dir path/to/your/tfrecords
```
You can also continue the training process from the snapshot by
```
python tools/train_shadownet.py --dataset_dir path/to/your/tfrecords --weights_path path/to/your/last/checkpoint
```
After several times of iteration you can check the log file in logs folder you are supposed to see the following contenent
![Training log](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/train_log.png)
The seq distance is computed by calculating the distance between two saparse tensor so the lower the accuracy value is the better the model performs.The train accuracy is computed by calculating the character-wise precision between the prediction and the ground truth so the higher the better the model performs.

During my experiment the `loss` drops as follows  
![Training loss](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/train_loss.png)
The `distance` between the ground truth and the prediction drops as follows  
![Sequence distance](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/seq_distance.png)

## Experiment
The accuracy during training process rises as follows  
![Training accuracy](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/training_accuracy.md)

## TODO
The model is trained on a subet of [Synth 90k](http://www.robots.ox.ac.uk/~vgg/data/text/). So i will train a new model on the whold dataset to get a more robust model.The crnn model needs large of training data to get a rubust model.



如何训练自己的中文模型
###### 1.通过爬虫或者其它手段,生成汉字字库  chinese_dict.txt
    比如本项目中用到的data/char_dict/chinese_dict.txt
###### 2.调用tools/establish_char_dict.py 根据字库生成字库对应的映射文件
    data/char_dict/char_dict.json
    data/char_dict/index_2_ord_map.json
    data/char_dict/ord_2_index_map.json
###### 3.调用TextRecognitionDataGenerator生成图片和文件标签labels.txt
    labels.txt中如下 
    data/Train/0/1000173.jpg 懦占边肩霄远已灶撑车
    data/Test/0/1000003.jpg 氟消悸稠退怖蚀刃竞镍
###### 4.调用tools/write_text_tfrecords.py 生成tfrecords记录
###### 5.根据tfrecord来训练模型    
  第一次直接调用 
  python tools/train_shadownet.py --dataset_dir data/tfrecords --train_num 4750000
  中途断了后,后续可以继续
  python tools/train_shadownet.py --dataset_dir data/tfrecords --train_num 4750000   --weights_path model/shadownet/shadownet_2018-07-06-09-20-32.ckpt-14000
                                                         
###### 6.用训练出来的模型来验证数据
  验证测试数据集
  python tools/test_shadownet.py --dataset_dir data/tfrecords --weights_path model/shadownet/shadownet_2018-06-28-12-50-28.ckpt-8000
  验证单独的图片
  python tools/demo_shadownet.py  --weights_path   model/shadownet/shadownet_2018-06-28-12-50-28.ckpt-8000 --image_path data/test_images/test_01.jpg
###### 7.部署到tensorflow service
    使用tf.saved_model.builder.SavedModelBuilder来导出模型
    导出的时候，通过add_meta_graph_and_variables的signature_def_map参数来定义相应的输入输出参数
    下载bazel-0.11.1-dist.zip与Tensorflow1.6对应的 service,编绎即可

如果报Saw a non-null label (index >= num_classes - 1) following a null label的话,
就要调整global_configuration/config.py中的CLASSES_NUMS,将它调大为比 data/char_dict/index_2_ord_map.json中最大序号大2
原因就是这个index_2_ord_map是所有汉字的分类,比如你有500个汉字,那么你的class_num就要比它大2, # 汉字的总个数 +  space + ctc blank label


查看显卡使用情况
C:\Program Files\NVIDIA Corporation\NVSMI  下nvidia-smi.exe


导出模型的样例

export_dir = ...
...
builder = tf.saved_model_builder.SavedModelBuilder(export_dir)
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph_and_variables(sess,
                                       [tag_constants.TRAINING],
                                       signature_def_map=foo_signatures,
                                       assets_collection=foo_assets)
...
# Add a second MetaGraphDef for inference.
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph([tag_constants.SERVING])
...
builder.save()


说明：
CTPN基于网上的一tensorflow ctpn实现，未做改动。
另外ctpn与crnn一起使用时，因为都是基于tensorflow，两个graph的问题没有解决，哪位朋友解决了麻烦跟我说一下(虽然实际部署都是分开部署的)
CRNN的代码基于以下链接的chinese版本实现的
https://github.com/MaybeShewill-CV/CRNN_Tensorflow
主要改动有：
####     1.原作者在write tfrecords时，直接将所有图片加载到内存的，因为公司机器性能差(16G内存=GTX1060 6G显卡)，所以修改这里，现在就算你10亿样本，也不怕内存不够了。
####     2.原作者直接用的RGB图来训练，但实际上三通道感觉没必要，第一导致样本数据更大，第二样本数据tfrecords大了2/3，于是我全改为单通道灰度的了。个人感觉像形文字单通道就够了，不关心颜色。
####     3.原作者在计算acc的时候，采用的是tf.nn.ctc_beam_search_decoder，即返回k个概率，会导致计算acc非常非常慢，可以换成ctc_greedy_decoder会快点，不过我这是直接改为2000个iteration才算一次，另外原作者打印的时候直接将iteration写为epoch,这里改了改。
####     4.原作者并未完全按CRNN论文来实现七层CNN,中间有几层都不一样，会导至感受野变小？具体影响如何我也不清楚，感觉CNN的理论支撑还是太弱了，大家都是经验。。。初学者，不懂，哪位同学可以给我解释下。。。
####     5.在小分类100以下汉字时，测试效果可以，但是大一点就无法收敛了，具体可能原因：样本太少，batch太小，LR要调大？

因为机器实在太慢了，折腾太久，每次试验一次一个把星期就没了，对人打击太大，就放弃折腾了，强烈建议上512GSSD+双卡GTX1080 以上才去尝试。


