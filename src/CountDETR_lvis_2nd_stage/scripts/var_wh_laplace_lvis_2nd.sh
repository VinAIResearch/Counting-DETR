#!/bin/bash 
python3.7 main.py \
--data_path ./FSCD_LVIS/ \
--output_dir ./outputs/var_wh_laplace_2nd \
--num_workers 0 \
--spatial_prior grid \
--batch_size 1 \
--no_aux_loss \
--num_query_pattern 1 \
--num_query_position 600 \
--resume ./pretrained_models/AnchorDETR_r50_c5.pth
# --resume ./pretrained_models/AnchorDETR_r50_c5.pth && \ 

# python3.7 infer.py \
# --data_path ./FSCD_LVIS/ \
# --output_dir ./outputs/var_wh_laplace_2nd/ \
# --num_workers 0 \
# --spatial_prior grid \
# --batch_size 1 \
# --no_aux_loss \
# --num_query_pattern 1 \
# --num_query_position 600 \
# --checkpoint_path ./outputs/var_wh_laplace_2nd/detr_retrain.pth && \

# python3.7 offline_lvis_evaluator.py --pred_json_path ./outputs/var_wh_laplace_2nd/predictions_test.json \
# --output_dir ./outputs/var_wh_laplace_2nd
