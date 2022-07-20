#!/bin/bash 
python3.8 generate_pseudo.py \
--dataset_file fscd_lvis_point \
--data_path ./FSCD_LVIS \
--output_dir ./outputs/fscd_lvis_1st_stage \
--num_workers 0 \
--spatial_prior defined \
--num_query_pattern 1 \
--vis_pseudo \
--resume ./outputs/fscd_lvis_1st_stage/checkpoint.pth
