<div align="center">
  <img src="/md/logo.png" width="503px"><br>
  
  <img src="/md/badge_python.svg"/>
  <img src="/md/badge_license.svg"/>
  <img src="/md/badge_docker.svg"/>
  <img src="/md/badge_cuda10.svg"/>
  <img src="/md/badge_cuda11.svg"/>
  
  SBD is an anchor free object detector that combines the advantages of YOLO and Centernet,
  
  Designed specifically to operate on edge devices, and is light, fast, and accurate
  
</div>

## Quick Start🚀
Installation
```bash
git clone https://github.com/inzapp/sbd

# local installation
cd sbd
python -m pip intall setup/requirements_cuda_xxx.txt

# docker installation
docker pull inzapp/sbd:cu118 # in case of cuda version 11.8
cd sbd/setup
./run_docker.sh
```

Train
```bash
python train.py --cfg cfg/cfg.yaml
```

Detect
```bash
cd checkpoint/model_name/
python ../../detect.py # detect with validation data path in cfg.yaml
python ../../detect.py --dataset train # detect with train data path in cfg.yaml
python ../../detect.py --path "/your/images/path/dir" # user defined image path dir
python ../../detect.py --path "/your/images/path/image.jpg" # one image detection
python ../../detect.py --path "/your/video/path.mp4" # realtime video detection
python ../../detect.py --path "rtsp://foo/bar" # rtsp stream realtime detection
python ../../detect.py --path "rtsp://user:passsword@foo/bar" # rtsp stream case need authentication
```

mAP calculation
```bash
cd checkpoint/model_name/
python ../../map.py # calculate mAP with validation data in cfg.yaml
python ../../map.py --cached --find-best-threshold # use cached csv, calculate mAP with best confidence threshold for each class
python ../../map.py --cached --find-best-threshold --dataset train # calculate mAP with train data in cfg.yaml
```

Auto label
```bash
cd checkpoint/model_name/
python ../../auto_label.py --path "/your/image/path/dir" # save yolo style label with predicted result
```

Multi GPU training
```yaml
# cfg/cfg.yaml
devices: [] # cpu training
devices: [0] # one gpu training with device index 0
devices: [2, 3] # 2 GPU training with device index 2, 3
devices: [0, 1, 2, 3] # 4 GPU training with device index 0, 1, 2, 3
```

ONNX export
```bash
./export.sh  # check opset version
```

## Introduction

SBD is designed as a single-scale model to prioritize fast inference and minimal post-processing.

It operates without anchors and does not involve mathematical operations like exp during post-processing, as seen in YOLO.

<img src="/md/structure.png" width="1000px"><br>

To leverage large-scale features, SBD use a lightweight FPN and decoupled head for efficient performance.

These features make SBD optimally suited for edge devices.

However, SBD is not ideal for highly complex and extensive datasets such as COCO.

The primary goal of SBD is to detect simple objects or objects in small-sized images with maximum efficiency in minimal time.
