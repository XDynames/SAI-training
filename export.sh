python ../detectron2/tools/deploy/caffe2_converter.py \
    --format onnx \
    --config-file configs/mask_rcnn_28x56.yaml \
    --run-eval \
    --output ./output/onnx/mask_rcnn_28x56/ \
    opts MODEL.WEIGHTS ./arabidopsis_weights.pth \
    MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION '(14, 14)'
