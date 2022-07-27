# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Train and eval functions used in main.py
"""
import json
import math
import os
import os.path as osp
import sys
from typing import Iterable

import cv2
import numpy as np
import torch
import util.misc as utils
import util.plot_utils as plot_utils


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("loss_wh", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("loss_giou", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    metric_logger.add_meter("grad_norm", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 100

    for ret in metric_logger.log_every(data_loader, print_freq, header):
        image = ret["image"].to(device)
        points = ret["points"].to(device)
        target = {
            "points": ret["points"].to(device),
            "whs": ret["whs"].to(device),
        }

        outputs = model(image, points)
        loss_dict = criterion(outputs, target)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_wh=loss_dict_reduced["loss_wh"])
        metric_logger.update(loss_giou=loss_dict_reduced["loss_giou"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    print_freq = 100
    for ret in metric_logger.log_every(data_loader, print_freq, header):
        image = ret["image"].to(device)
        points = ret["points"].to(device)
        target = ({"points": ret["points"].to(device), "whs": ret["whs"].to(device)})
        outputs = model(image, points)
        loss_dict = criterion(outputs, target)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_wh=loss_dict_reduced["loss_wh"])
        metric_logger.update(loss_giou=loss_dict_reduced["loss_giou"])
        # results = postprocessors['bbox'](outputs, orig_target_sizes)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


@torch.no_grad()
def generate_pseudo_label(model, criterion, postprocessors, data_loader, split, device, output_dir, is_vis=False):
    image_output_dir = osp.join(output_dir, "vis_pseudo_label")
    os.makedirs(image_output_dir, exist_ok=True)
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    pseudo_annotation = dict()
    pseudo_annotation["categories"] = [{"name": "fg", "id": 1}]
    pseudo_annotation["images"] = list()
    pseudo_annotation["annotations"] = list()

    print_freq = 100
    img_id = 1
    anno_id = 1
    for ret in metric_logger.log_every(data_loader, print_freq, header):
        image = ret["image"].to(device)
        old_im_id = ret["im_id"].item()
        img_name = str(old_im_id) + ".jpg"
        points = ret["points"].to(device)
        outputs = model(image, points)
        orig_target_sizes = ret["orig_size"].to(device)
        pred_whs = outputs["pred_wh"]

        points = points.squeeze(0).detach().cpu().numpy()
        pred_whs = torch.squeeze(pred_whs).detach().cpu().numpy()
        orig_target_sizes = orig_target_sizes.squeeze(0).detach().cpu().numpy()
        pred_whs[:, 0] *= orig_target_sizes[0]
        pred_whs[:, 1] *= orig_target_sizes[1]
        points[:, 0] *= orig_target_sizes[0]
        points[:, 1] *= orig_target_sizes[1]
        for point, wh in zip(points, pred_whs):
            x_cen, y_cen = point
            w, h = wh
            anno = {
                "id": anno_id,
                "image_id": img_id,
                "area": int(w * h),
                "bbox": [int(x_cen), int(y_cen), int(w), int(h)],
                "category_id": 1,
                "iscrowd": 0,
            }
            pseudo_annotation["annotations"].append(anno)
            anno_id += 1

        img_size = ret["orig_size"].detach().numpy()
        width, height = img_size[0][0], img_size[0][1]
        img_info = {
            "id": img_id,
            "file_name": img_name,
            "height": int(height),
            "width": int(width),
        }
        pseudo_annotation["images"].append(img_info)
        img_id += 1
    print(output_dir)
    with open(os.path.join(output_dir, "pseudo_bbox_" + split + ".json"), "w") as handle:
        json.dump(pseudo_annotation, handle)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, device, output_dir, is_vis=False):
    image_output_dir = osp.join(output_dir, "test_set")
    os.makedirs(image_output_dir, exist_ok=True)
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    pseudo_annotation = dict()
    pseudo_annotation["categories"] = [{"name": "fg", "id": 1}]
    pseudo_annotation["images"] = list()
    pseudo_annotation["annotations"] = list()

    print_freq = 100
    # img_id = 1
    anno_id = 1
    for ret in metric_logger.log_every(data_loader, print_freq, header):
        image = ret["image"].to(device)
        ori_id = ret["ori_id"].item()
        img_name = str(ori_id) + ".jpg"

        points = ret["points"].to(device)
        print(points)
        # points = ret["anchor_points"].to(device)
        outputs = model(image, points)
        orig_target_sizes = ret["orig_size"].to(device)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        pred_points, pred_boxes = results[0]["points"], results[0]["boxes"]
        for pred_box in pred_boxes:
            # x_min, y_min, w, h = pred_box
            x_min, y_min, x_max, y_max = pred_box
            w, h = x_max - x_min, y_max - y_min
            anno = {
                "id": anno_id,
                # "image_id": img_id,
                # "image_id": image_id,
                "image_id": ori_id,
                "area": int(w * h),
                "bbox": [int(x_min), int(y_min), int(w), int(h)],
                "category_id": 1,
                "iscrowd": 0,
            }
            pseudo_annotation["annotations"].append(anno)
            anno_id += 1
        if is_vis:
            # image_path = osp.join('FSC147', "images_384_VarV2", img_name)
            image_path = osp.join(image_output_dir, img_name)
            img = cv2.imread(image_path)
            np_array = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            old_h, old_w, _ = img.shape
            new_h, new_w, _ = np_array.shape
            point_ratio = np.array([old_w / new_w, old_h / new_h], dtype=np.float32)
            pred_points = pred_points * point_ratio
            vis_image = plot_utils.draw_pseudo_label(img, pred_points, pred_boxes)
            output_image_path = osp.join(image_output_dir, img_name)
            cv2.imwrite(output_image_path, vis_image)
        img_size = ret["orig_size"].detach().numpy()
        height, width = img_size[0][0], img_size[0][1]
        img_info = {
            # "id": image_id,
            "id": ori_id,
            "file_name": "None",
            "height": int(height),
            "width": int(width),
        }
        pseudo_annotation["images"].append(img_info)
        # img_id += 1

    with open(os.path.join(output_dir, "pseudo_test_anchor_detr_v3.json"), "w") as handle:
        json.dump(pseudo_annotation, handle)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats
