python demo/demo.py \
    --config-file configs/mask_rcnn_final_v4_cloud.yaml \
<<<<<<< HEAD
    --output output-Hv_GABA_ABA/ \
    --input datasets/Hv_GABA_ABA/ \
=======
    --output output-human-samples/ \
    --input datasets/barley/human_benchmark/images/ \
    --annotations datasets/barley/human_benchmark/annotations/GroundTruth.json \
>>>>>>> cd09bd4... Initial
    --confidence-threshold 0.5 \
    --opts MODEL.WEIGHTS ./barley_weights_human_trail.pth
