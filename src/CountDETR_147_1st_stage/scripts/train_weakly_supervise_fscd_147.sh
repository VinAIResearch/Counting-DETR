#!/bin/bash 
python3.7 main.py --dataset_file fscd_147 \
--data_path ./FSC147/ \
--output_dir ./outputs/fscd_147_1st_stage \
--num_workers 0 \
--spatial_prior defined \
--num_query_pattern 1 \
--resume ./pretrained_models/AnchorDETR_r50_c5.pth && \

python3.7 main.py --dataset_file fscd_147_point \
--data_path ./FSC147/ \
--output_dir ./outputs/fscd_147_1st_stage/ \
--num_workers 0 \
--spatial_prior defined \
--generate_pseudo_label \
--num_query_pattern 1 \
--vis_pseudo \
--resume ./outputs/fscd_147_1st_stage/checkpoint.pth \
