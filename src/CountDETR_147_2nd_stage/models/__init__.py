# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from .anchor_center import build_anchor_center
from .anchor_detr import build
from .centerness import build_centerness


def build_model(args):
    return build(args)


def build_centerness_model(args):
    return build_centerness(args)


def build_anchor_center_model(args):
    return build_anchor_center(args)
