python demo/demo.py \
    --config-file configs/mask_rcnn_arabidopsis_final_v4_cloud.yaml \
    --output output_demo_arabidopsis \
    --input datasets/arabidopsis/human_benchmark/images \
    --annotations datasets/arabidopsis/human_benchmark/annotations/GroundTruth.json \
    --confidence-threshold 0.5 \
    --opts MODEL.WEIGHTS ./arabidopsis_weights.pth