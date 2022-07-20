#!/bin/bash 
python3.7 infer.py \
--data_path ./FSC147/ \
--output_dir ./outputs/var_wh_laplace_600/ \
--num_workers 0 \
--spatial_prior grid \
--batch_size 1 \
--no_aux_loss \
--num_query_pattern 1 \
--num_query_position 600 \
--checkpoint_path ./outputs/var_wh_laplace_600/detr_retrain.pth

# python3.7 evaluate_val_and_test.py --input_folder ./outputs/var_wh_laplace_600/
