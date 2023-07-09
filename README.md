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

Training
```bash
python train.py --cfg cfg/cfg.yaml
```

Detecting
```bash
cd checkpoint/model_name/model_type/
python ../../../detect.py # need hardware monitor or X11 forward for CLI system
python ../../../detect.py --video "/your/video/path.mp4" # realtime video detecting
```

mAP
```bash
cd checkpoint/model_name/model_type/
python ../../../map.py
python ../../../map.py --conf 0.1 --cached # fast calculate using saved csv files
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
