#!/bin/bash

set -e
set -x

number_of_processes=8
data_dir="./data/mvtec/"
output_dir="./output/pixpro_mvtec"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
python -m torch.distributed.launch --master_port 12350 --nproc_per_node=${number_of_processes} \
    main_pretrain.py \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --cache-mode no \
    --crop 0.08 \
    --aug BYOL \
    --dataset MVTec \
    --batch-size 64 \
    --image-size 224 \
    --resized_image_size 256 \
    --crop_patch_size 256 \
    --crop_patch_stride 256 \
    \
    --model PixPro \
    --arch resnet18 \
    --head-type early_return \
    \
    --optimizer lars \
    --base-lr 0.1 \
    --weight-decay 0.0003 \
    --warmup-epoch 5 \
    --epochs 400 \
    --amp-opt-level O1 \
    \
    --save-freq 100 \
    --auto-resume \
    \
    --pixpro-p 2 \
    --pixpro-momentum 0.99 \
    --pixpro-pos-ratio 0.7 \
    --pixpro-transform-layer 1 \
    --pixpro-ins-loss-weight 0 \
    \
    --mvtec_category all 
