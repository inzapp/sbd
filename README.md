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
cd sbd

# local installation
python -m pip intall setup/requirements_cuda_xxx.txt

# docker installation
cd setup/
docker build --no-cache -t sbd -f Dockerfile.cuxxx .
./run_docker.sh
```

Train
```bash
python train.py --cfg cfg/cfg.yaml
```

Detect
```bash
# need hardware monitor or X11 forward for CLI system
cd checkpoint/model_name/model_type/
python ../../../detect.py # detect with validation data path in cfg.yaml
python ../../../detect.py --dataset train # detect with train data path in cfg.yaml
python ../../../detect.py --path "/your/images/path/dir" # user defined image path dir
python ../../../detect.py --path "/your/images/path/image.jpg" # one image detection
python ../../../detect.py --path "/your/video/path.mp4" # realtime video detection
python ../../../detect.py --path "rtsp://foo/bar" # rtsp stream realtime detection
python ../../../detect.py --path "rtsp://user:passsword@foo/bar" # case need authentication
```

mAP
```bash
cd checkpoint/model_name/model_type/
python ../../../map.py # calculate mAP with validation data in cfg.yaml
python ../../../map.py --dataset train # calculate mAP with train data in cfg.yaml
python ../../../map.py --conf 0.1 --iou 0.6 --cached # fast calculation using cached csv file for --conf, --iou. must run map.py at least once
```

Auto label
```bash
cd checkpoint/model_name/model_type/
python ../../../auto_label.py --path "/your/image/path/dir" --conf 0.3 # save label with predicted result
```

## Introduction
SBD is a light and fast object detector

This can be particularly useful in edge devices

Because there are several layers that are not supported by most edge devices, SBD uses a backbone that is only used as a light csp block consisting of a vanilla convolution layer

Because fast processing is the primary purpose of SBD, post processing is also very concise and fast

- [x] Totally anchor free
- [x] Low memory required
- [x] ALE loss with convex IoU loss function
- [x] No mathematical operations for post processing
- [x] Customized learning rate scheduler
- [x] Lightweight CSP block
- [x] Lightweight FPN block
- [x] Virtual anchor training
- [x] Multi scale training
- [x] Various output resolution
- [x] Heatmap ignore trick
- [x] Support SBD-P6 model
- [x] Multi GPU training
- [x] Fast training loop