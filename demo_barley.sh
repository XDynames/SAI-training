python demo/demo.py \
    --config-file configs/mask_rcnn_final_v4_cloud.yaml \
    --output output-human-samples-3/ \
    --input datasets/datasets/barley/human_benchmark/ \
    --annotations datasets/datasets/barley/stoma/annotations/val.json \
    --confidence-threshold 0.0 \
    --opts MODEL.WEIGHTS ./barley_weights_human_trail.pth