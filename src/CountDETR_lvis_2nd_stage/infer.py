# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from data import build_test_dataset
from models import build_model
import json

torch.backends.cudnn.enabled = False

@torch.no_grad()
def infer(model, criterion, data_loader, device, output_dir, split="test"):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    predictions = dict()
    predictions["categories"] = [{'name': 'fg', 'id': 1}]
    predictions["images"] = list()
    predictions["annotations"] = list()

    anno_id = 1        
    
    image_id = 1

    for ret in metric_logger.log_every(data_loader, 100, header):
        image = ret['image'].to(device)
        rects = ret['ex_rects'].to(device)
        # all_outputs = model(image, points=sampled_points, rects=rects)
        all_outputs = model(image, rects=rects)
        outputs = all_outputs[0]; ref_points = all_outputs[1] 
        
        targets = [{
            "boxes": ret['boxes'].squeeze(0).to(device),
            "labels": ret['labels'].squeeze(0).to(device)
        }]
        ori_shape = ret["orig_size"].detach().cpu().numpy()
        ori_h, ori_w = ori_shape[0]
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        #metric_logger.update(loss=loss_value)
        #metric_logger.update(class_error=loss_dict_reduced['class_error'])
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        prob = out_logits.sigmoid()
        object_prob = prob[...,0]
        threshold = 0.5
        obj_pos = torch.where(object_prob >= threshold)
        
        pred_scores = object_prob[obj_pos[0],obj_pos[1]]
        pred_boxes = out_bbox[obj_pos[0],obj_pos[1]]
        ref_points = ref_points[obj_pos[0],obj_pos[1]]
        pred_points = ref_points.detach().cpu().numpy()
        pred_points[...,0] *= ori_w
        pred_points[...,1] *= ori_h

        pred_boxes = pred_boxes.detach().cpu().numpy()
        pred_boxes[...,0] *= ori_w; pred_boxes[...,1] *= ori_h
        pred_boxes[...,2] *= ori_w; pred_boxes[...,3] *= ori_h

        
        ori_h, ori_w = ret["orig_size"].numpy()[0]
        img_info = {
            "id": image_id, 
            "height": int(ori_h),
            "width": int(ori_w),
            "file_name": "None",
        }

        for pred_score, pred_box, pred_point in zip(pred_scores, pred_boxes, pred_points):
            x_cen, y_cen, w, h = pred_box
            x_ref, y_ref = pred_point
            anno = {
                "id": anno_id, 
                "image_id": image_id, 
                "area": int(w*h), 
                "bbox": [int(x_cen), int(y_cen), int(w), int(h)], 
                "category_id": 1,
                "score": float(pred_score), 
                "point": [int(x_ref), int(y_ref)], 

            }
            predictions["annotations"].append(anno)
            anno_id += 1
        predictions["images"].append(img_info)

        image_id += 1

    print(image_id)
    
    with open(os.path.join(output_dir, "predictions_"+split+".json"), "w") as handle:
        json.dump(predictions, handle)

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_args_parser():
    parser = argparse.ArgumentParser('AnchorDETR Detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=[], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr_drop', default=20, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--vis_pseudo', action='store_true', help="Visualize generated pseudo label")

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', default=True,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--num_feature_levels', default=1, type=int, help='number of feature levels')
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0., type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_query_position', default=300, type=int,
                        help="Number of query positions")
    parser.add_argument('--num_query_pattern', default=3, type=int,
                        help="Number of query patterns")
    parser.add_argument('--spatial_prior', default='learned', choices=['learned', 'grid', 'defined'],
                        type=str,help="Number of query patterns")
    parser.add_argument('--attention_type',
                        # default='nn.MultiheadAttention',
                        default="RCDA",
                        choices=['RCDA', 'nn.MultiheadAttention'],
                        type=str,help="Type of attention module")
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_points', default=2, type=float,
                        help="points coefficient in the matching cost")

    # * Matcher
    parser.add_argument('--cost_class', default=2, type=float,
                    help="Class coefficient in the matching cost")
    parser.add_argument('--cost_bbox', default=5, type=float,
                    help="L1 box coefficient in the matching cost")
    parser.add_argument('--cost_giou',default=2, type=float,
                    help="giou box coefficient in the matching cost")
                    
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--point_loss_coef', default=2, type=float)
    parser.add_argument('--variance_loss_coef', default=2, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='fscd_147_pseudo', choices=['fscd_147_pseudo', 'fscd_147_test'])
    parser.add_argument('--data_path', default=None, type=str, required=True)
    
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default=None, required=True, 
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', default=False, action='store_true', help='whether to resume from last checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')

    parser.add_argument('--generate_pseudo_label', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--scale_factor', default=32, type=int, help="scale factor of image")

    parser.add_argument('--checkpoint_path', required=True,  help='checkpoint to test')
    parser.add_argument('--use_predefined_points', action='store_true', help="to use predefine anchor points")
    
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
    print('number of params:', n_parameters)

    dataset_test = build_test_dataset(args=args)
    data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out


    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    pretrained_dict = checkpoint['model']
    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(pretrained_dict, strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))

    infer(model, criterion,
                data_loader_test, device, args.output_dir, split="test")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('AnchorDETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
