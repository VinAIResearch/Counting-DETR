import os
import shutil
from os.path import join, isdir
import numpy as np
import argparse
import pickle as pkl

def analyze_result(img_res, output_dir):
    vis_res_folder = join(output_dir, "vis_res")
    assert isdir(vis_res_folder), "Results must be visualized"

    for img_re in img_res:
        pred_count = img_re["count_pred"]
        gt_count = img_re["count_gt"]
        img_re["diff"] = gt_count - pred_count

    low_ap_img_folder = join(vis_res_folder, "low_ap")
    os.makedirs(low_ap_img_folder, exist_ok=True)
    sorted_results = sorted(img_res, key=lambda d: d['ap']) 
    img_names = [each_img_res["img_name"] for idx, each_img_res in enumerate(sorted_results) if idx > 1000]
    for img_name in img_names:
        old_path = join(vis_res_folder, img_name)
        new_path = join(low_ap_img_folder, img_name)
        shutil.copyfile(old_path, new_path)

    diff_img_folder = join(vis_res_folder, "diff")
    os.makedirs(diff_img_folder, exist_ok=True)
    sorted_results = sorted(img_res, key=lambda d: d['diff']) 
    img_names = [each_img_res["img_name"] for idx, each_img_res in enumerate(sorted_results) if idx < 300]
    for img_name in img_names:
        old_path = join(vis_res_folder, img_name)
        new_path = join(diff_img_folder, img_name)
        shutil.copyfile(old_path, new_path)

    return 

def get_args_parser():
    parser = argparse.ArgumentParser('AnchorDETR Detector', add_help=False)
    parser.add_argument('-p', '--pred_pkl', default="./outputs/fscd_147_gnn/each_img_infor.pkl", type=str)
    parser.add_argument('-o', '--output_dir', default="./outputs/fscd_147_gnn/", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args_parser()
    pred_pkl = args.pred_pkl
    output_dir = args.output_dir
    with open(pred_pkl, "rb") as handle:
        img_res = pkl.load(handle)
    analyze_result(img_res, output_dir)
