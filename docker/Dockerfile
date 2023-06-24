FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y libgeos-dev libglib2.0-0 libsm6 libxext6 libxrender-dev

RUN python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

COPY ./src /src
WORKDIR /src

RUN pip install -e .