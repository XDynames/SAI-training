python demo/demo.py \
    --config-file configs/mask_rcnn_barley.yaml \
    --output output-samples/ \
    --input datasets/barley/human_benchmark/images/ \
    --annotations datasets/barley/human_benchmark/annotations/GroundTruth.json \
    --confidence-threshold 0.5 \
    --opts MODEL.WEIGHTS ./barley_weights_human_trail.pth
