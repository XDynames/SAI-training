python demo/demo.py \
    --config-file configs/mask_rcnn_R_50_FPN.yaml \
    --output output_demo_v1_all/ \
    --input datasets/stoma/val/ \
    --confidence-threshold 0.7 \
    --opts MODEL.WEIGHTS output_v1/mask/R_50/model_final.pth