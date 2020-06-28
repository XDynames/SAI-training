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

To train the end-to-end Mask R-CNN model first convert the xml annotations into coco format json with

```
python datasets/create_cocofied_annotations.py
```

## Demo

First download the trained model from [this link](https://cloudstor.aarnet.edu.au/plus/s/sfkLBWae8bmal6s). Then run

```
python demo/demo.py \
    --config-file configs/faster_rcnn_R_50_FPN.yaml \
    --input datasets/stoma_detection/val/ \
    --confidence-threshold 0.5 \
    --opts MODEL.WEIGHTS R_50.pth
```