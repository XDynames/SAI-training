python demo/demo.py \
    --config-file configs/mask_rcnn_barley.yaml \
    --output output_demo_barley/ \
    --input datasets/barley/stoma/human \
    --annotations datasets/barley/stoma/annotations/GroundTruth.json \
    --confidence-threshold 0.5 \
    --gpu \
    --opts MODEL.WEIGHTS ./barley_weights.pth
