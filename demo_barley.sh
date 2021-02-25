python demo/demo.py \
    --config-file configs/mask_rcnn_final_v4_cloud.yaml \
    --output output-Hv_GABA_ABA/ \
    --input datasets/Hv_GABA_ABA/ \
    --confidence-threshold 0.5 \
    --opts MODEL.WEIGHTS ./barley_weights_human_trail.pth
