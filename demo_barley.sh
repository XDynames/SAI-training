python demo/demo.py \
    --config-file configs/mask_rcnn_barley.yaml \
    --output output_Hv_L2D_22Nov_L2D_1mMGABA \
    --input datasets/barley/Hv_L2D/22Nov_L2D_1mMGABA \
    --confidence-threshold 0.5 \
    --gpu \
    --opts MODEL.WEIGHTS ./barley_weights.pth