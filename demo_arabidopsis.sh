python3 demo/demo.py \
    --config-file configs/mask_rcnn_arabidopsis.yaml \
    --output output_human_samples \
    --input datasets/arabidopsis/human_benchmark \
    --confidence-threshold 0.5 \
    --opts MODEL.WEIGHTS ./arabidopsis_weights.pth