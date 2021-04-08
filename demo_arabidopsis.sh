python3 demo/demo.py \
    --config-file configs/mask_rcnn_arabidopsis.yaml \
    --output output_light_3h \
    --input datasets/light_3h \
    --confidence-threshold 0.5 \
    --opts MODEL.WEIGHTS ./arabidopsis_weights.pth