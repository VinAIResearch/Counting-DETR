import argparse
import json
import os
from pathlib import Path

import torch
import util.misc as utils
from data.fsc147 import FSC147Dataset
from engine import evaluate, train_one_epoch
from models import build_model
from torch.utils.data import DataLoader


torch.backends.cudnn.enabled = False


def config_parser():
    parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
    parser.add_argument("-dp", "--data_path", type=str, default="./FSC147/", help="Path to the FSC147 dataset")
    parser.add_argument("-o", "--output_dir", type=str, default="./outputs/anchor_detr", help="/Path/to/output/logs/")
    parser.add_argument(
        "-ts",
        "--test-split",
        type=str,
        default="val",
        choices=["train", "test", "val"],
        help="what data split to evaluate on on",
    )

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

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )

    # * Matcher
    parser.add_argument("--cost_class", default=2, type=float, help="Class coefficient in the matching cost")
    parser.add_argument("--cost_bbox", default=5, type=float, help="L1 box coefficient in the matching cost")

    parser.add_argument("--cost_giou", default=2, type=float, help="giou box coefficient in the matching cost")

    parser.add_argument("--chamfer_point_cost", default=1)
    parser.add_argument("--chamfer_giou_cost", default=1)

    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--cls_loss_coef", default=2, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)
    parser.add_argument("--variance_loss_coef", default=2, type=float)

    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--auto_resume", default=False, action="store_true", help="whether to resume from last checkpoint"
    )
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--cache_mode", default=False, action="store_true", help="whether to cache images on memory")

    args = parser.parse_args()
    return args


def main(args):

    os.makedirs(args.output_dir, exist_ok=True)
    anchor_detr, criterion, _ = build_model(args)
    anchor_detr.cuda()

    dataset_train = FSC147Dataset(args, split="train")
    dataset_val = FSC147Dataset(args, split="val")

    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True)

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
                for n, p in anchor_detr.named_parameters()
                if not match_name_keywords(n, args.lr_backbone_names)
                and not match_name_keywords(n, args.lr_linear_proj_names)
                and p.requires_grad
            ],
            "lr": args.lr,
        },
        {
            "params": [
                p
                for n, p in anchor_detr.named_parameters()
                if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                p
                for n, p in anchor_detr.named_parameters()
                if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad
            ],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        anchor_detr.detr.load_state_dict(checkpoint["model"])

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        pretrained_model = checkpoint["model"]
        pretrained_dict = {
            k: v
            for k, v in pretrained_model.items()
            if k in anchor_detr.state_dict() and "transformer.pattern." not in k
        }

        missing_keys, unexpected_keys = anchor_detr.load_state_dict(pretrained_dict, strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith("total_params") or k.endswith("total_ops"))]
        if len(missing_keys) > 0:
            print("Missing Keys: {}".format(missing_keys))
        if len(unexpected_keys) > 0:
            print("Unexpected Keys: {}".format(unexpected_keys))

    print("Start training")
    print(args.output_dir)
    import time

    start = time.time()
    output_dir = Path(args.output_dir)
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            anchor_detr, criterion, dataloader_train, optimizer, args.device, epoch, args.clip_max_norm
        )
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / "detr_retrain.pth"]
        if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
            checkpoint_paths.append(output_dir / f"detr_retrain_{epoch:04}.pth")
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master(
                {
                    "model": anchor_detr.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                },
                checkpoint_path,
            )
        print("Evaluate ...")
        # test_stats = evaluate(anchor_detr,criterion,dataloader_val,args.device)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            #  **{f'test_{k}': v for k, v in test_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "detr_retrain.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    end = time.time()
    print("time: ", end - start)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    args = config_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
