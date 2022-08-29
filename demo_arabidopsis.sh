python demo/demo.py \
    --config-file configs/mask_rcnn_arabidopsis.yaml \
    --output output_At \
    --input datasets/arabidopsis/stoma/images \
    --confidence-threshold 0.5 \
    --gpu \
    --opts MODEL.WEIGHTS ./arabidopsis_weights.pth