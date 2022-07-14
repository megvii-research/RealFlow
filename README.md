# [ECCV 2022] RealFlow: EM-based Realistic Optical Flow Dataset Generation from Videos

## Requirements
- torch>=1.8.1
- torchvision>=0.9.1
- opencv-python>=4.5.2
- timm>=0.4.5
- cupy>=5.0.0
- numpy>=1.15.0

## Rendered Datasets
Download part of [RF-AB]()

Other datasets. Coming soon.

## Render New Data
Download the pretrained DPT model from [here](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt) and pretrained RAFT model from [here](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing)

Download [KITTI multi-view](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php) Datasets.
You can run the following command to render RF-Ktrain:
```shell
python RealFlow.py
```

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
