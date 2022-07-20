#!/bin/bash 
# python3.8 main.py \
# --data_path ./FSCD_LVIS/ \
# --output_dir ./outputs/finetune_v2 \
# --num_workers 0 \
# --spatial_prior grid \
# --batch_size 1 \
# --no_aux_loss \
# --num_query_pattern 1 \
# --num_query_position 300 \
# --start_epoch 30 \
# --epochs 60 \
# --resume ./outputs/finetune/detr_retrain.pth && \ 

python3.8 infer.py \
--data_path ./FSCD_LVIS/ \
--output_dir ./outputs/finetune_v2/ \
--num_workers 0 \
--spatial_prior grid \
--batch_size 1 \
--no_aux_loss \
--num_query_pattern 1 \
--num_query_position 300 \
--checkpoint_path ./outputs/finetune_v2/detr_retrain.pth && \

python3.8 offline_lvis_evaluator.py --pred_json_path ./outputs/finetune_v2/predictions_test.json \
--output_dir ./outputs/finetune_v2
