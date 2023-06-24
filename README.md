# StomaAI: Detection and Measurement
## About
StomaAI (SAI) aims to automate the measuring process of stomate pore features from microscope images that are commonly used in plant physiological research. Previously researchers would spend days manually measuring pore's lengths, widths and opening areas. Now this can be done in minutes. SAI is a joint collaboration between The University of Adelaide's Australian Institute for Machine Learning and Plant Energy Biology ARC Center of Excellence.

## Docker
For convenience we provide a docker file that can be built to run our code in.
To use this ensure you have installed both [Docker](https://docs.docker.com/engine/install/) and [Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
You can modify the file in `/docker/compose.yml` to adjust the configuration file being used, data locations and where to write outputs.
Running: `docker compose --project-directory ./docker/ up -d --build` will build and start the training container.
To monitor training you can use tensorboard specifying the log directory as the output folder of the run.

## Installation
Ensure you have libgeos installed: `sudo apt-get install libgeos-dev`

Install the appropriate versions of [Pytorch](https://pytorch.org/get-started/locally/) and [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
It is very important that the versions requested by the Detectron package are respected.

Run `bash setup.sh` to download the validation image set, associated ground truth annotations, model weights and install the StomaAI package with its dependencies.
These can then used to both reproduce the reported evaluation scores or generate visualisations of measurements on the two validation sets provided.

## Inference Evaluation
After installing you can check the evaluation metrics reported for both the models by running:
```
bash evaluate_arabidopsis.sh
bash evaluate_barley.sh
bash evaluate_combined.sh # This does barley and arabidopsis sequentially
```
These should agree with the following:
| Model                                                        | BB mAP | Mask mAP | Keypoint mAP | mAP open | mAP closed | weights                                                                     |
| ------------------------------------------------------------ | :----: | :------: | :----------: | :------: | :--------: | --------------------------------------------------------------------------- |
| [Barley](configs/mask_rcnn_barley.yaml)                      | 80.67  |  69.06   |    77.44     |  86.31   |   68.57    | [download](https://cloudstor.aarnet.edu.au/plus/s/KWFjWBLlE18n9M9/download) |
| [Arabidopsis](configs/mask_rcnn_arabidopsis.yaml)            | 74.67  |  43.74   |    43.89     |  53.99   |   33.78    | [download](https://cloudstor.aarnet.edu.au/plus/s/iLB4PwuKqjbdSWg/download) |
| [Combined - Barley](configs/mask_rcnn_barley.yaml)           | 80.85  |  67.72   |    78.65     |  88.66   |   68.64    | [download](https://cloudstor.aarnet.edu.au/plus/s/EQMljoS9YLvpHtS/download) |
| [Combined - Arabidopsis](configs/mask_rcnn_arabidopsis.yaml) | 75.70  |  46.19   |    50.87     |  56.822  |   44.92    | [download](https://cloudstor.aarnet.edu.au/plus/s/EQMljoS9YLvpHtS/download) |

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
Then arrange your data into the folder structure bellow:
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

## Referencing
If you use SAI as part of your work please cite us:
```
@article{https://doi.org/10.1111/nph.18765,
author = {Sai, Na and Bockman, James Paul and Chen, Hao and Watson-Haigh, Nathan and Xu, Bo and Feng, Xueying and Piechatzek, Adriane and Shen, Chunhua and Gilliham, Matthew},
title = {StomaAI: an efficient and user-friendly tool for measurement of stomatal pores and density using deep computer vision},
journal = {New Phytologist},
volume = {n/a},
number = {n/a},
pages = {},
keywords = {applied deep learning, computer vision, convolutional neural network, phenotyping, stomata},
doi = {https://doi.org/10.1111/nph.18765},
url = {https://nph.onlinelibrary.wiley.com/doi/abs/10.1111/nph.18765},
eprint = {https://nph.onlinelibrary.wiley.com/doi/pdf/10.1111/nph.18765},
}
```
