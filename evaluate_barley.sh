python3 tools/train_net.py \
    --eval-only \
    --resume \
    --dataset-dir datasets/barley/ \
    --config-file configs/mask_rcnn_barley.yaml  \
    MODEL.WEIGHTS ./barley_weights.pth \
    OUTPUT_DIR './output_barley' \