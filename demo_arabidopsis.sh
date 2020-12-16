python demo/demo.py \
    --config-file configs/mask_rcnn_arabidopsis_final_v4_cloud.yaml \
    --output output_demo_arabidopsis-streamlit/ \
    --input datasets/datasets/arabidopsis/human_benchmark \
    --annotations datasets/datasets/arabidopsis/stoma/annotations/val.json \
    --confidence-threshold 0.0 \
    --opts MODEL.WEIGHTS ./arabidopsis_v4_weights.pth