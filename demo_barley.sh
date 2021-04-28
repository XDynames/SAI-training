python demo/demo.py \
    --config-file configs/mask_rcnn_barley.yaml \
    --output output-samples/ \
    --input datasets/barley/stoma/images/ \
    --confidence-threshold 0.5 \
    --gpu \
    --opts MODEL.WEIGHTS ./barley_weights_human_trail.pth
    