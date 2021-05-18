python3 tools/train_net.py \
    --eval-only \
    --resume \
    --dataset-dir datasets/arabidopsis/ \
    --config-file configs/mask_rcnn_arabidopsis.yaml  \
    MODEL.WEIGHTS ./arabidopsis.pth \
    OUTPUT_DIR './output' \