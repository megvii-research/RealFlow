# [ECCV 2022 Oral] RealFlow: EM-based Realistic Optical Flow Dataset Generation from Videos ([Paper](https://arxiv.org/pdf/2207.11075.pdf))

<h4 align="center">Yunhui Han$^1$, Kunming Luo$^2$, Ao Luo$^2$, Jiangyu Liu$^2$, Haoqiang Fan$^2$, Guiming Luo$^1$, Shuaicheng Liu$^3$</center>
<h4 align="center">1. Tsinghua University, 2. Megvii Research</center>
<h4 align="center">3. University of Electronic Science and Technology of China</center>


## Abstract
Obtaining the ground truth labels from a video is challenging since the manual annotation of pixel-wise flow labels is prohibitively expensive and laborious. Besides, existing approaches try to adapt the trained model on synthetic datasets to authentic videos, which inevitably suffers from domain discrepancy and hinders the performance for realworld applications. To solve these problems, we propose RealFlow, an Expectation-Maximization based framework that can create large-scale optical flow datasets directly from any unlabeled realistic videos. Specifically, we first estimate optical flow between a pair of video frames, and then synthesize a new image from this pair based on the predicted flow. Thus the new image pairs and their corresponding flows can be regarded as a new training set. Besides, we design a Realistic Image Pair Rendering (RIPR) module that adopts softmax splatting and bi-directional hole filling techniques to alleviate the artifacts of the image synthesis. In the E-step, RIPR renders new images to create a large quantity of training data. In the M-step, we utilize the generated training data to train an optical flow network, which can be used to estimate optical flows in the next E-step. During the iterative learning steps, the capability of the flow network is gradually improved, so is the accuracy of the flow, as well as the quality of the synthesized dataset. Experimental results show that RealFlow outperforms previous dataset generation methods by a considerably large margin. Moreover, based on the generated dataset, our approach achieves state-of-the-art performance on two standard benchmarks compared with both supervised and unsupervised optical flow methods

## Motivation
![motivation](https://user-images.githubusercontent.com/1344482/180913272-d8e1af87-b305-4beb-b067-ff29ce53a56d.JPG)

Top: previous methods use synthetic motion to produce training pairs. Bottom: we propose to construct training pairs with realistic motion labels from the real-world video sequence. We estimate optical flow between two frames as the training label and synthesize a ‘New Image 2’. Both the new view and flow labels are refined iteratively in the EM-based framework for mutual improvements.

## Requirements
- torch>=1.8.1
- torchvision>=0.9.1
- opencv-python>=4.5.2
- timm>=0.4.5
- cupy>=5.0.0
- numpy>=1.15.0

## Rendered Datasets
![results](https://user-images.githubusercontent.com/1344482/180913871-cbbce758-8b03-46b5-b3a4-b07f0b229f82.JPG)

Download RF-AB datasets (~40G) on BaiduCloud [here](https://pan.baidu.com/s/17lBkQbYOR2TcoFvs3Zj93Q?pwd=w19u) 

Our pretrained RF-AB/RF-DAVIS model 

## Render New Data
Download the pretrained DPT model from [here](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt) and pretrained RAFT C+T model from [here](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing)

Download [KITTI multi-view](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php) Datasets.
You can run the following command to render RF-Ktrain:
```shell
python RealFlow.py
```
You can also download ALOV and BDD100k from their official website to render RF-AB. Using utils/video2img.py to capture pictures.


You can simply render a new pair using:
```shell
python demo.py
```

## Acknowledgements
Part of the code is adapted from previous works:
- [RAFT](https://github.com/princeton-vl/RAFT)
- [DPT](https://github.com/isl-org/DPT)
- [Softmax Splatting](https://github.com/sniklaus/softmax-splatting)


We thank all the authors for their awesome repos.
