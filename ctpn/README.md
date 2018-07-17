# ctpn network

Learning how ctpn works in scene text detection area referring to [ctpn](https://github.com/eragonruan/text-detection-ctpn)<br>
Debug and comment!

demo 
python ./ctpn/demo.py

linux平台安装
    cd lib/utils
    chmod +x make.sh
    ./make.sh


windows平台安装
    cd lib/utils
    把cython_nms.pyx，gpu_nms.pyx三个文件中所有的 np.int_t 改为 np.intp_t
    cython bbox.pyx
    cython cython_nms.pyx
    cython gpu_nms.pyx
    python setup_win.py build_ext --inplace
    xcopy /e /y /r /i  .\utils\*.*  .\
    rm -rf build  
    rm -rf utils

如果报参数类型错误，修改以下代码为
_nms((&(*__Pyx_BufPtrStrided1d(__pyx_t_5numpy_int32_t *, __pyx_pybuffernd_keep.rcbuffer->pybuffer.buf, __pyx_t_10, __pyx_pybuffernd_keep.diminfo[0].strides))), (&__pyx_v_num_out), (&(*__Pyx_BufPtrStrided2d(__pyx_t_5numpy_float32_t *, __pyx_pybuffernd_sorted_dets.rcbuffer->pybuffer.buf, __pyx_t_12, __pyx_pybuffernd_sorted_dets.diminfo[0].strides, __pyx_t_13, __pyx_pybuffernd_sorted_dets.diminfo[1].strides))), __pyx_v_boxes_num, __pyx_v_boxes_dim, __pyx_t_14, __pyx_v_device_id);

_nms((&(*__Pyx_BufPtrStrided1d(int *, __pyx_pybuffernd_keep.rcbuffer->pybuffer.buf, __pyx_t_10, __pyx_pybuffernd_keep.diminfo[0].strides))), (&__pyx_v_num_out), (&(*__Pyx_BufPtrStrided2d(__pyx_t_5numpy_float32_t *, __pyx_pybuffernd_sorted_dets.rcbuffer->pybuffer.buf, __pyx_t_12, __pyx_pybuffernd_sorted_dets.diminfo[0].strides, __pyx_t_13, __pyx_pybuffernd_sorted_dets.diminfo[1].strides))), __pyx_v_boxes_num, __pyx_v_boxes_dim, __pyx_t_14, __pyx_v_device_id);

如果报参数int* long long的错误，就把cython_nms.pyx，gpu_nms.pyx三个文件中所有的 np.int_t 改为 np.intp_t

如果报rc.exe找不到的话，看下是不是安装了最新的win10 sdk,配置一下这两个环境变量
set  WindowsSDK_ExecutablePath_x64=C:\Program Files (x86)\Windows Kits\10\bin\10.0.17134.0\x64
set  WindowsSDK_ExecutablePath_x86=C:\Program Files (x86)\Windows Kits\10\bin\10.0.17134.0\x86
set  PATH=%WindowsSDK_ExecutablePath_x64%;%PATH%

当然了，也可以直接修改vcvar**.bat


# parameters
there are some parameters you may need to modify according to your requirement, you can find them in ctpn/text.yml
- USE_GPU_NMS # whether to use nms implemented in cuda or not
- DETECT_MODE # H represents horizontal mode, O represents oriented mode, default is H
- checkpoints_path # the model I provided is in checkpoints/, if you train the model by yourself,it will be saved in output/
***
# demo
- download the checkpoints from release, unzip it in checkpoints/
- put your images in data/demo, the results will be saved in data/results, and run demo in the root 
```shell
python ./ctpn/demo.py
```
***
# training
## prepare data
- First, download the pre-trained model of VGG net and put it in data/pretrain/VGG_imagenet.npy. you can download it from [google drive](https://drive.google.com/open?id=0B_WmJoEtfQhDRl82b1dJTjB2ZGc) or [baidu yun](https://pan.baidu.com/s/1kUNTl1l). 
- Second, prepare the training data as referred in paper, or you can download the data I prepared from [google drive](https://drive.google.com/open?id=0B_WmJoEtfGhDRl82b1dJTjB2ZGc) or [baidu yun](https://pan.baidu.com/s/1kUNTl1l). Or you can prepare your own data according to the following steps. 
- Modify the path and gt_path in prepare_training_data/split_label.py according to your dataset. And run
```shell
cd lib/prepare_training_data
python split_label.py
```
- it will generate the prepared data in current folder, and then run
```shell
python ToVoc.py
```
- to convert the prepared training data into voc format. It will generate a folder named TEXTVOC. move this folder to data/ and then run
```shell
cd ../../data
ln -s TEXTVOC VOCdevkit2007
```
## train 
Simplely run
```shell
python ./ctpn/train_net.py
```
- you can modify some hyper parameters in ctpn/text.yml, or just used the parameters I set.
- The model I provided in checkpoints is trained on GTX1070 for 50k iters.
- If you are using cuda nms, it takes about 0.2s per iter. So it will takes about 2.5 hours to finished 50k iterations.
***
# roadmap
- [x] cython nms
- [x] cuda nms
- [x] python2/python3 compatblity
- [x] tensorflow1.3
- [x] delete useless code
- [x] loss function as referred in paper
- [x] oriented text connector
- [x] BLSTM
- [ ] side refinement
***
