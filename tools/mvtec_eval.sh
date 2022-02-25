#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

python3 mvtec_eval.py \
    --category all \
    --density gde \
    --note "" \
    --imagenet_resnet False \
    --qualitative False \
