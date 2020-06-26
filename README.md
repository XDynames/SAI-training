# Stoma Detection and Measurement

## Installation

First install detectron2.

Then run

```
python setup.py build develop
```

## Training

```
python tools/train_net.py --resume --config-file configs/faster_rcnn_R_50_FPN.yaml
```