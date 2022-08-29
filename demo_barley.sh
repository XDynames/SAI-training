python demo/demo.py \
    --config-file configs/mask_rcnn_barley.yaml \
    --output output_Hv \
    --input datasets/barley/stoma/images \
    --confidence-threshold 0.5 \
    --gpu \
    --opts MODEL.WEIGHTS ./barley_weights.pth