python tools/train_net.py \
    --eval-only \
    --resume \
    --dataset-dir datasets/barley/ \
    --config-file configs/mask_rcnn_barley.yaml  \
    MODEL.WEIGHTS ./combined_weights.pth \
    OUTPUT_DIR './output_combined_barley' \

python tools/train_net.py \
    --eval-only \
    --resume \
    --dataset-dir datasets/arabidopsis/ \
    --config-file configs/mask_rcnn_arabidopsis.yaml  \
    MODEL.WEIGHTS ./combined_weights.pth \
    OUTPUT_DIR './output_combined_arabidopsis' \