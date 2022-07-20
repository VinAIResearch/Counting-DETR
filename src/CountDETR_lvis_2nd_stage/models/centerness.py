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
AnchorDETR model and criterion classes.
"""
from numpy.core.numeric import indices
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_centerness_matcher
import copy

class CountRegressor(nn.Module):
    def __init__(self, input_channels,pool='mean'):
        super(CountRegressor, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 224, 5, padding=3),
            nn.LeakyReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(224, 196, 5, padding=3),
            nn.LeakyReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.LeakyReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, 1),
            nn.LeakyReLU(),
        )

    def forward(self, im):
        num_sample =  im.shape[0]
        if num_sample == 1:
            output = self.regressor(im.squeeze(0))
            if self.pool == 'mean':
                output = torch.mean(output, dim=(0),keepdim=True)  
                return output
            elif self.pool == 'max':
                output, _ = torch.max(output, 0,keepdim=True)
                return output
        else:
            for i in range(0,num_sample):
                output = self.regressor(im[i])
                if self.pool == 'mean':
                    output = torch.mean(output, dim=(0),keepdim=True)
                elif self.pool == 'max':
                    output, _ = torch.max(output, 0,keepdim=True)
                if i == 0:
                    Output = output
                else:
                    Output = torch.cat((Output,output),dim=0)
            return Output

class CenternessPredictor(nn.Module):
    """ This is the AnchorDETR module that performs object detection """

    def __init__(self, backbone, centerness_head, num_feature_levels, ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
        """
        super().__init__()
        self.centerness_head = centerness_head
        hidden_dim = 256

        self.num_feature_levels = num_feature_levels
        self.feat_aggr_type = 3
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                if _ == 0:
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                else:
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        
        # self.aggr_input_proj = nn.ModuleList([
        #         nn.Sequential(
        #             nn.Conv2d(backbone.num_channels[0] * 3, hidden_dim, kernel_size=1),
        #             nn.GroupNorm(32, hidden_dim),
        #         )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # for proj in self.aggr_input_proj:
        #     nn.init.xavier_uniform_(proj[0].weight, gain=1)
        #     nn.init.constant_(proj[0].bias, 0)

    def forward(self, samples: NestedTensor, rects):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        # features = self.backbone(samples, rects)
        features = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src).unsqueeze(1))
            # srcs.append(self.aggr_input_proj[l](src).unsqueeze(1))
            masks.append(mask)
            assert mask is not None

        srcs = torch.cat(srcs, dim=1)
        output_map = self.centerness_head(srcs)
        return output_map


class CriterionCenterNess(nn.Module):
    def __init__(self, weight_dict):
        super().__init__()
        self.weight_dict = weight_dict
        self.loss = nn.MSELoss()
    
    def forward(self,outputs,targets):
        '''
        Loss computation
        outputs: dict of tensors
        targets: list of dict
        '''
        centerness_maps = torch.cat([t["centerness_map"] for t in targets], dim=0)
        outputs = F.interpolate(outputs, centerness_maps.shape)
        outputs = outputs.squeeze(0).squeeze(0)        
        mask_loss = self.loss(outputs, centerness_maps)
        losses = {"mask_loss": mask_loss}
        return losses

class TestPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        new_h, new_w = target_sizes[0][0], target_sizes[0][1]
        outputs = nn.Sigmoid()(outputs)
        new_outputs = F.interpolate(outputs, (new_h, new_w))
        return new_outputs.squeeze(0)

def build_centerness(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    matcher = build_centerness_matcher(args)
    weight_dict = {'mask_loss': 1.0,}
    # TODO this is a hack

    device = torch.device(args.device)

    backbone = build_backbone(args)
    centerness_head = CountRegressor(256)

    model = CenternessPredictor(
        backbone,
        centerness_head,
        num_feature_levels=args.num_feature_levels,
    )
    
    criterion = CriterionCenterNess(weight_dict=weight_dict)
    criterion.to(device)
    postprocessors = {'bbox': TestPostProcess()}
    return model, criterion, postprocessors

def build_centerness_head(args):
    return CountRegressor(256)

