FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

RUN apt-get update && apt-get install -y libgeos-dev libglib2.0-0 libsm6 libxext6 libxrender-dev

RUN python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

COPY ./src /src

RUN bash setup.sh