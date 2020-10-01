python demo/demo.py \
    --config-file configs/mask_rcnn_R_50_FPN_arabidopsis_higher_low_cosine_cloud.yaml \
    --output output_demo_arabidopsis/ \
    --input datasets/arabidopsis/stoma/val \
    --annotations datasets/arabidopsis/stoma/annotations/val.json \
    --confidence-threshold 0.5 \
    --opts MODEL.WEIGHTS ./arabidopsis_weights.pth