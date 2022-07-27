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
from .fscd_lvis import build_fscd_lvis, build_fscd_lvis_points, build_fscd_lvis_test


def build_dataset(image_set, args):
    if image_set == "test":
        return build_fscd_lvis_test(args, image_set)
    if args.dataset_file == "fscd_lvis":
        return build_fscd_lvis(args, image_set)
    elif args.dataset_file == "fscd_lvis_point":
        return build_fscd_lvis_points(args, image_set)
    else:
        raise ValueError(f"dataset {args.dataset_file} not supported")
