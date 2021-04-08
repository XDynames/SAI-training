FROM xdynames/detectron2-container:latest

RUN sudo apt-get update && sudo apt-get install -y libgeos-dev libglib2.0-0 

WORKDIR /home/appusr/sai

COPY . .

RUN ls

RUN  pip install --user -e .

CMD [ "bash train.sh" ]