#!/bin/bash
for id in "$@"
do
    echo "Retrieving Pore $id"
    python tools/retrieve_pore.py \
        --pore-id "$id" \
        --image-folder output_demo_arabidopsis \
        --annotations datasets/arabidopsis/human_benchmark/annotations/GroundTruth.json
done


#--image-folder datasets/arabidopsis/human_benchmark/images \