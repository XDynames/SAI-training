python ./tools/export.py \
    --dataset-dir datasets/datasets/barley/ \
    --config-file configs/mask_rcnn_final_v4_cloud.yaml \
    --output ./output/onnx/mask_rcnn_v4/ \
    MODEL.WEIGHTS ./barley_weights_human_trail.pth