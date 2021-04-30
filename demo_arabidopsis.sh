python3 demo/demo.py \
    --config-file configs/mask_rcnn_arabidopsis.yaml \
    --output output_at_human_trail \
    --input datasets/arabidopsis/human_benchmark \
    --annotations datasets/arabidopsis/stoma/annotations/val.json \
    --confidence-threshold 0.5 \
    --gpu \
    --opts MODEL.WEIGHTS ./arabidopsis_weights.pth