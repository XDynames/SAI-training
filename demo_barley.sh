python demo/demo.py \
    --config-file configs/mask_rcnn_final_v4_cloud.yaml \
    --output output-human-samples/ \
    --input datasets/barley/stoma/val \
    --annotations datasets/barley/stoma/annotations/val.json \
    --confidence-threshold 0.5 \
    --opts MODEL.WEIGHTS ./barley_weights.pth