python3 demo/demo.py \
    --config-file configs/mask_rcnn_arabidopsis.yaml \
    --output output_210421 \
    --input datasets/arabidopsis/210421 \
    --confidence-threshold 0.5 \
    --gpu \
    --opts MODEL.WEIGHTS ./arabidopsis_weights.pth