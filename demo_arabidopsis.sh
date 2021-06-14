python3 demo/demo.py \
    --config-file configs/mask_rcnn_arabidopsis.yaml \
    --output output_demo_arabidopsis \
    --input datasets/arabidopsis/stoma/human/ \
    --annotations datasets/arabidopsis/stoma/annotations/GroundTruth.json \
    --confidence-threshold 0.5 \
    --gpu \
    --opts MODEL.WEIGHTS ./arabidopsis_weights.pth