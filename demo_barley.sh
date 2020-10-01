python demo/demo.py \
    --config-file configs/mask_rcnn_R_50_FPN_highest_low_cosine_cloud.yaml \
    --output output_demo_barley/ \
    --input datasets/barley/stoma/val \
    --annotations datasets/barley/stoma/annotations/val.json \
    --confidence-threshold 0.5 \
    --opts MODEL.WEIGHTS ./barley_weights.pth