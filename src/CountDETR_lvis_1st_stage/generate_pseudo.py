# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import argparse
import json
import os
import os.path as osp
import random
from pathlib import Path

import numpy as np
import torch
import util.misc as utils
from datasets import build_dataset
from models import build_model
from torch.utils.data import DataLoader


@torch.no_grad()
def generate_pseudo_label(model, criterion, postprocessors, data_loader, split, device, output_dir, is_vis=False):
    image_output_dir = osp.join(output_dir, "vis_pseudo_label")
    os.makedirs(image_output_dir, exist_ok=True)
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    cxcywh_pseudo_annotation = dict()
    cxcywh_pseudo_annotation["categories"] = [{"name": "fg", "id": 1}]
    cxcywh_pseudo_annotation["images"] = list()
    cxcywh_pseudo_annotation["annotations"] = list()

    xywh_pseudo_annotation = dict()
    xywh_pseudo_annotation["categories"] = [{"name": "fg", "id": 1}]
    xywh_pseudo_annotation["images"] = list()
    xywh_pseudo_annotation["annotations"] = list()

    print_freq = 100
    img_id = 1
    anno_id = 1
    for ret in metric_logger.log_every(data_loader, print_freq, header):
        image = ret["image"].to(device)
        img_name = ret["img_name"][0]
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

            xywh_anno = {
                "id": anno_id,
                "image_id": img_id,
                "area": int(w * h),
                "bbox": [int(x_cen - w / 2), int(y_cen - h / 2), int(w), int(h)],
                "category_id": 1,
                "iscrowd": 0,
            }

            cxcywh_pseudo_annotation["annotations"].append(anno)
            xywh_pseudo_annotation["annotations"].append(xywh_anno)
            anno_id += 1

        img_size = ret["orig_size"].detach().numpy()
        width, height = img_size[0][0], img_size[0][1]
        img_info = {
            "id": img_id,
            "file_name": img_name,
            "height": int(height),
            "width": int(width),
        }
        cxcywh_pseudo_annotation["images"].append(img_info)
        xywh_pseudo_annotation["images"].append(img_info)

        img_id += 1
    print(output_dir)
    with open(os.path.join(output_dir, "pseudo_lvis_" + split + "_cxcywh.json"), "w") as handle:
        json.dump(cxcywh_pseudo_annotation, handle)

    with open(os.path.join(output_dir, "pseudo_lvis_" + split + "_xywh.json"), "w") as handle:
        json.dump(xywh_pseudo_annotation, handle)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


def get_args_parser():
    parser = argparse.ArgumentParser("AnchorDETR Detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone_names", default=["backbone"], type=str, nargs="+")
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--lr_linear_proj_names", default=[], type=str, nargs="+")
    parser.add_argument("--lr_linear_proj_mult", default=0.1, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--lr_drop", default=20, type=int)
    parser.add_argument("--lr_drop_epochs", default=None, type=int, nargs="+")
    parser.add_argument("--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm")

    parser.add_argument("--sgd", action="store_true")
    parser.add_argument("--vis_pseudo", action="store_true", help="Visualize generated pseudo label")

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )

    # * Backbone
    parser.add_argument("--backbone", default="resnet50", type=str, help="Name of the convolutional backbone to use")
    parser.add_argument(
        "--dilation",
        default=True,
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument("--num_feature_levels", default=1, type=int, help="number of feature levels")
    # * Transformer
    parser.add_argument("--enc_layers", default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument("--dec_layers", default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument(
        "--dim_feedforward",
        default=1024,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim", default=256, type=int, help="Size of the embeddings (dimension of the transformer)"
    )
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout applied in the transformer")
    parser.add_argument(
        "--nheads", default=8, type=int, help="Number of attention heads inside the transformer's attentions"
    )
    parser.add_argument("--num_query_position", default=300, type=int, help="Number of query positions")
    parser.add_argument("--num_query_pattern", default=3, type=int, help="Number of query patterns")
    parser.add_argument(
        "--spatial_prior",
        default="learned",
        choices=["learned", "grid", "defined"],
        type=str,
        help="Number of query patterns",
    )
    parser.add_argument(
        "--attention_type",
        # default='nn.MultiheadAttention',
        default="RCDA",
        choices=["RCDA", "nn.MultiheadAttention"],
        type=str,
        help="Type of attention module",
    )
    # * Segmentation
    parser.add_argument("--masks", action="store_true", help="Train segmentation head if the flag is provided")

    # * Matcher
    parser.add_argument("--set_cost_class", default=2, type=float, help="Class coefficient in the matching cost")
    parser.add_argument("--set_cost_bbox", default=5, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument("--set_cost_giou", default=2, type=float, help="giou box coefficient in the matching cost")
    parser.add_argument("--set_cost_points", default=2, type=float, help="points coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--cls_loss_coef", default=2, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)
    parser.add_argument("--point_loss_coef", default=2, type=float)

    # dataset parameters
    parser.add_argument(
        "--dataset_file",
        default="fscd_147",
        choices=["fscd_147", "fscd_lvis", "fscd_147_point", "fscd_147_test", "fscd_lvis_point"],
    )
    parser.add_argument("--data_path", default=None, type=str, required=True)

    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--output_dir", default=None, required=True, help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--auto_resume", default=False, action="store_true", help="whether to resume from last checkpoint"
    )
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--cache_mode", default=False, action="store_true", help="whether to cache images on memory")
    parser.add_argument("--scale_factor", default=32, type=int, help="scale factor of image")
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and p.requires_grad
            ],
            "lr": args.lr,
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
    ]

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = torch.load(args.resume, map_location="cpu")

    pretrained_dict = checkpoint["model"]
    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(pretrained_dict, strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith("total_params") or k.endswith("total_ops"))]
    if len(missing_keys) > 0:
        print("Missing Keys: {}".format(missing_keys))
    if len(unexpected_keys) > 0:
        print("Unexpected Keys: {}".format(unexpected_keys))

    dataset_train = build_dataset(image_set="train", args=args)
    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    generate_pseudo_label(
        model, criterion, postprocessors, data_loader_train, "train", device, args.output_dir, args.vis_pseudo
    )

    dataset_test = build_dataset(image_set="test", args=args)
    data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
    generate_pseudo_label(
        model, criterion, postprocessors, data_loader_test, "test", device, args.output_dir, args.vis_pseudo
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AnchorDETR training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
