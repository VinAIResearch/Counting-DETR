#!/bin/bash 
python3.8 offline_lvis_evaluator.py --pred_json_path ./outputs/FamnetRR/predictions_test.json \
--output_dir ./outputs/FamnetRR
# python3.8 offline_lvis_evaluator.py --pred_json_path ./outputs/AttentionRPN-RRbox/predictions_test.json \
# --output_dir ./outputs/AttentionRPN-RRbox &&
# python3.8 offline_lvis_evaluator.py --pred_json_path ./outputs/Fsdetview-rrbbox/predictions_test.json \
# --output_dir ./outputs/Fsdetview-rrbbox &&
# python3.8 offline_lvis_evaluator.py --pred_json_path ./outputs/AttentionRPN-pseudobox/predictions_test.json \
# --output_dir ./outputs/AttentionRPN-pseudobox &&
# python3.8 offline_lvis_evaluator.py --pred_json_path ./outputs/Fsdetview-pseudobbox/predictions_test.json \
# --output_dir ./outputs/Fsdet    view-pseudobbox && 
# python3.8 offline_lvis_evaluator.py --pred_json_path ./outputs/var_wh_laplace_2nd/predictions_test.json \
# --output_dir ./outputs/var_wh_laplace_2nd
