docker run --rm \
    --shm-size 2G \
    --gpus all \
    -v /mnt/data/stomata/barley_v3:/data \
    -v /home/james/Documents/SAI-training/output:/output/ \
    -v /home/james/Documents/SAI-training/configs:/configs/ \
    sai-training python ./demo/demo.py \
    --config-file /configs/mask_rcnn_barley_2.yaml \
    --output /output/demo_barley_v3 \
    --input /data/stoma/images \
    --confidence-threshold 0.5 \
    --gpu \
    --opts MODEL.WEIGHTS /output/barley_v3/model_final.pth