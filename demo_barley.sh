python demo/demo.py \
    --config-file configs/mask_rcnn_barley.yaml \
    --output demo_output_barley/ \
    --input datasets/barley/stoma/val/ \
    --confidence-threshold 0.5 \
    --gpu \
    --opts MODEL.WEIGHTS ./barley_weights.pth
