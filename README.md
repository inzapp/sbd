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
./export.sh  # just copy and paste your model path for exporting
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
- [x] Lightweight FPN head
- [x] Virtual anchor training
- [x] Multi scale training
- [x] Various output resolution
- [x] Heatmap ignore trick
- [x] Support SBD-P6 model
- [x] Multi GPU training
- [x] Fast training loop

## Model
SBD provides two types of models: one output layer model and multi layer output model

And you can choose the output resolution scale by changing model_type in cfg.yaml file

```yaml
model_type: m1p2 # medium backbone one output layer model with pyramid scale 2
```

The description of the model_type is as follows

m : backbone type, available backbones : n(nano), s(small), m(medium), l(large), x(x-large)

1 : 1 for one output layer model or m for multi layer model

p : constant character for naming rule

2 : pyramid scale value, 2 ~ 6(p6 only) is available

available model type examples) n1p5, nmp3, s1p2, smp4, m1p3, mmp2, l1p4, lmp2, x1p5, xmp3

You can change the resolution of the output layer by changing the pyramid scale

Max output layer resolution is divided by 2^p of input resolution

When using the multi layer model, the number of output layers depends on the paramid scale

Through the FPN head until the paramid scale is reached,

output layers are added one by one from the lowest resolution layer

For better understanding, take a model with model_type m1p2 as an example

<img src="/md/model.png" width="850px"/>

if you want to use p6 model, modify cfg.yaml as below

```yaml
p6_model: True
```

p6 model has addtional downscaling block and use 64 strides so input resolution must be multiple of 64

if p6_model: False in cfg.yaml, 32 strides is used by default
