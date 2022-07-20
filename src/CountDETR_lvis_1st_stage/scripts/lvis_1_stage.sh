#!/bin/bash 
python3.7 main.py --dataset_file fscd_lvis \
--data_path ./FSCD_LVIS \
--output_dir ./outputs/fscd_lvis_1st_stage \
--num_workers 0 \
--spatial_prior defined \
--num_query_pattern 1 \
--resume ./pretrained_models/AnchorDETR_r50_c5.pth && \

python3.7 generate_pseudo.py \
--dataset_file fscd_lvis \
--data_path ./FSCD_LVIS \
--output_dir ./outputs/fscd_lvis_1st_stage \
--num_workers 0 \
--spatial_prior defined \
--generate_pseudo_label \
--num_query_pattern 1 \
--vis_pseudo \
--resume ./outputs/fscd_lvis_1st_stage/checkpoint.pth \
