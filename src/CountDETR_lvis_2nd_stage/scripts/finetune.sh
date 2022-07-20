#!/bin/bash 
python3.8 main.py \
--data_path ./FSCD_LVIS/ \
--output_dir ./outputs/finetune \
--num_workers 0 \
--spatial_prior grid \
--batch_size 1 \
--no_aux_loss \
--num_query_pattern 1 \
--num_query_position 300 \
--resume ./pretrained_models/AnchorDETR_r50_c5.pth && \ 

python3.8 infer.py \
--data_path ./FSCD_LVIS/ \
--output_dir ./outputs/finetune/ \
--num_workers 0 \
--spatial_prior grid \
--batch_size 1 \
--no_aux_loss \
--num_query_pattern 1 \
--num_query_position 300 \
--checkpoint_path ./outputs/finetune/detr_retrain.pth && \

python3.8 offline_lvis_evaluator.py --pred_json_path ./outputs/finetune/predictions_test.json \
--output_dir ./outputs/finetune
