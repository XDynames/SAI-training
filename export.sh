python ./tools/caffe2_converter.py \
    --dataset-dir datasets/barley/ \
    --format onnx \
    --config-file configs/mask_rcnn_final_v4_cloud.yaml \
    --run-eval \
    --output ./output/onnx/mask_rcnn_v4/ \
    MODEL.WEIGHTS ./barley_weights_human_trail.pth