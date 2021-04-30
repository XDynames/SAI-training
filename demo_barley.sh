python demo/demo.py \
    --config-file configs/mask_rcnn_barley.yaml \
    --output output-hv-human-trail/ \
    --input datasets/barley/stoma/human_trail/ \
    --annotations datasets/barley/stoma/annotations/val.json \
    --confidence-threshold 0.5 \
    --gpu \
    --opts MODEL.WEIGHTS ./barley_weights.pth
