python demo/demo.py \
    --config-file configs/mask_rcnn_arabidopsis_cloud.yaml \
    --output output_demo_arabidopsis/ \
    --input datasets/stoma/val \
    --confidence-threshold 0.5 \
    --opts MODEL.WEIGHTS ./arabidopsis_1.pth