python ./tools/caffe2_converter.py \
    --format onnx \
    --config-file configs/mask_rcnn_28x56.yaml \
    --run-eval \
    --output ./output/onnx/mask_rcnn_28x56/ \
    MODEL.WEIGHTS ./arabidopsis_weights.pth