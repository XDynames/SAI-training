python3 demo/demo.py \
    --config-file configs/mask_rcnn_arabidopsis.yaml \
    --output demo_output_arabidopsis \
    --input datasets/arabidopsis/val \
    --confidence-threshold 0.5 \
    --gpu \
    --opts MODEL.WEIGHTS ./arabidopsis_weights.pth