CUDA_VISIBLE_DEVICES="1,0" python tools/train_net.py \
    --num-gpus 2 \
    --resume \
    --config-file configs/mask_rcnn_28x56.yaml