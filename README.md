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
|-- finalized_data/
    |-- image_annotation
        Original_imgs
    stoma/
    |-- images/  (ln -s datasets/finalized_data/Original_imgs datasets/stoma/images)
        annotations/ (run python datasets/create_cocofied_annotations.py)
        val/
    stoma_detection/ (data from the py-faster-rcnn folder)
    |-- annotations
        train
        val
    

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

### Detection

First download the trained model from [this link](https://cloudstor.aarnet.edu.au/plus/s/sfkLBWae8bmal6s). Then run

```
python demo/demo.py \
    --config-file configs/faster_rcnn_R_50_FPN.yaml \
    --input datasets/stoma_detection/val/ \
    --confidence-threshold 0.5 \
    --opts MODEL.WEIGHTS R_50.pth
```

### End-to-End

First download the trained model from [this link](https://cloudstor.aarnet.edu.au/plus/s/1AJlUYksklDsDZH). Then run

```
python demo/demo.py \
    --config-file configs/mask_rcnn_R_50_FPN.yaml \
    --input datasets/stoma/val/ \
    --confidence-threshold 0.5 \
    --opts MODEL.WEIGHTS mask_R_50.pth
```