python3 tools/train_net.py \
    --eval-only \
    --resume \
    --dataset-dir datasets/arabidopsis/ \
    --config-file configs/mask_rcnn_arabidopsis.yaml  \
    MODEL.WEIGHTS ./arabidopsis_weights.pth \
    OUTPUT_DIR './arabidopsis_output' \