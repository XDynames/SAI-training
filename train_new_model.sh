python3 tools/train_net.py \
    --resume \
    --dataset-dir datasets/arabidopsis/ \
    --config-file configs/mask_rcnn_arabidopsis.yaml  \
    OUTPUT_DIR './output' \
    MODEL.WEIGHTS ./arabidopsis_weights.pth