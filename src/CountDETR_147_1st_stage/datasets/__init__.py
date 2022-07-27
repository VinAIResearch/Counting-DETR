# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .fscd_147 import build_fscd_147, build_fscd_147_points, build_fscd_test


def build_dataset(image_set, args):
    if args.dataset_file == "fscd_147":
        return build_fscd_147(args, image_set)
    elif args.dataset_file == "fscd_147_point":
        return build_fscd_147_points(args, image_set)
    elif args.dataset_file == "fscd_147_test":
        return build_fscd_test(args, image_set)
    else:
        raise ValueError(f"dataset {args.dataset_file} not supported")
