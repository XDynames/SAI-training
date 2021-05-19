# Stoma Detection and Measurement

## Installation
Ensure you have libgeos installed: `sudo apt-get install libgeos-dev`
Install the appropriate versions of [Pytorch](https://pytorch.org/get-started/locally/) and [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

Run `bash setup.sh` to download the validation image set, associated ground truth annotations and model weights.
These can then used to both reproduce the reported evaluation scores or generate visualisations of measurements on the two validation sets provided.

## Inference Evaluation
After installing you can check the evaluation metrics reported for both the models by running:
```
bash evaluate_arabidopsis.sh
bash evaluate_barley.sh
```
These should agree with the following:
| Model       | BB AP | Mask AP | Keypoint AP | AP open | AP closed | weights |
| ----------- | :---------: | :---------: | :---------: | :-----: | :-------: | ------- |
| [Barley](configs/mask_rcnn_barley.yaml) | 80.67 | 69.06 | 77.44 | 86.31 | 68.57 | [download](https://cloudstor.aarnet.edu.au/plus/s/KWFjWBLlE18n9M9) |
| [Arabidopsis](configs/mask_rcnn_arabidopsis.yaml)  | 74.67 |   43.74  |  43.89   | 53.99 | 33.78 | [download](https://cloudstor.aarnet.edu.au/plus/s/iLB4PwuKqjbdSWg) |

## Inference Demonstration
After installing you can visualise the measurements produced by each of the models on their validation datasets by running:
```
bash demo_arabidopsis.sh
bash demo_barley.sh
```
These will create a new folder each, `output_demo_arabidopsis` and `output_demo_barley`, containing the visualised predictions.

## Training
First convert the xml annotations into coco format json using:
```
python datasets/create_cocofied_annotations.py
```
Then arrange your datae into the folder structure bellow:
```
|-- datasets/
    |-- <species name>
        |-- stoma/
            |-- annotations/
                |-- val.json
                |-- train.json
            |-- images/
                |-- All images
            |-- val/
                |-- Validation images
```
Modify the `train_new_model.sh` script by changing the `--dataset-dir` to point to your data.


