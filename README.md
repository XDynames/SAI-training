# Stoma Detection and Measurement

## Installation

First install detectron2.

Then run

```
python setup.py build develop
```

Link datasets in the corresponding folder:

```
stoma/
tools/
datasets/
--  stoma/
    --  annotations/
    --  train/
    --  val/
```

## Training

```
python tools/train_net.py --resume --config-file configs/faster_rcnn_R_50_FPN.yaml
```