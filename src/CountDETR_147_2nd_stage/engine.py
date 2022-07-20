import os
import util.misc as utils
import math
import sys
import torch
import numpy as np
import json
import cv2
import torch.nn as nn

torch.backends.cudnn.enabled = False
def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, max_norm = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for ret in metric_logger.log_every(data_loader, print_freq, header):
        image = ret['image'].to(device)
        if 'sampled_points' in ret.keys():
            sampled_points = ret['sampled_points'].squeeze(0).numpy()
        else:
            sampled_points = None
        rects = ret['ex_rects'].to(device)
        targets = [{
            "boxes": ret['boxes'].squeeze(0).to(device),
            "labels": ret['labels'].squeeze(0).to(device)
        }]

        outputs, ref_points = model(image, points=sampled_points, rects=rects)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        #print(loss_dict_reduced_unscaled)
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
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        #metric_logger.update(loss=loss_value)
        #metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        #metric_logger.update(grad_norm=grad_total_norm)
        #metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def infer(model, criterion, data_loader, device, output_dir):
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
    
    for ret in metric_logger.log_every(data_loader, 100, header):
        image = ret['image'].to(device)
        if 'sampled_points' in ret.keys():
           sampled_points = ret['sampled_points'].squeeze(0).numpy()
        else: 
            sampled_points = None
        rects = ret['exemplar_boxes'].to(device)
        all_outputs = model(image, points=sampled_points, rects=rects)
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
        
        num_obj = obj_pos[1].shape[0]
        np_prob = object_prob.detach().cpu().numpy()[0]

        sorted_prob = np.sort(np_prob)
        sorted_prob = np.flip(sorted_prob)
        if num_obj*2-1 < 900:
            threshold = sorted_prob[num_obj*2-1]
        else:
            threshold = 0.0
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

        
        image_id = ret["image_id"].item()
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

    with open(os.path.join(output_dir, "predictions.json"), "w") as handle:
        json.dump(predictions, handle)

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_anchor_center(model, criterion, data_loader, optimizer, device, epoch, max_norm = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for ret in metric_logger.log_every(data_loader, print_freq, header):
        image = ret['image'].to(device)
        orig_size = ret['orig_size']
        if 'sampled_points' in ret.keys():
            sampled_points = ret['sampled_points'].squeeze(0).numpy()
        else:
            sampled_points = None
        rects = ret['ex_rects'].to(device)
        all_outputs = model(image, orig_size=orig_size)
        targets = [{
            "boxes": ret['boxes'].squeeze(0).to(device),
            "labels": ret['labels'].squeeze(0).to(device),
            "centerness_map": ret["centerness_map"].squeeze(0).to(device), 
        }]

        ori_outputs, centerness_map, ref_points = all_outputs[0], all_outputs[1], all_outputs[2]
        outputs = (ori_outputs, centerness_map)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        #print(loss_dict_reduced_unscaled)
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
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        #metric_logger.update(loss=loss_value)
        #metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        #metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        #metric_logger.update(grad_norm=grad_total_norm)
        #metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        #metric_logger.update(class_error=loss_dict_reduced['class_error'])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_anchor_center(model,criterion,data_loader, device):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    for ret in metric_logger.log_every(data_loader, 100, header):
        image = ret['image'].to(device)
        orig_size = ret['orig_size']

        if 'sampled_points' in ret.keys():
           sampled_points = ret['sampled_points'].squeeze(0).numpy()
        else: sampled_points = None
        targets = [{
            "boxes": ret['boxes'].squeeze(0).to(device),
            "labels": ret['labels'].squeeze(0).to(device), 
            "centerness_map": ret["centerness_map"].squeeze(0).to(device), 

        }]

        all_outputs = model(image, orig_size=orig_size)
        ori_outputs, centerness_map, ref_points = all_outputs[0], all_outputs[1], all_outputs[2]
        outputs = (ori_outputs, centerness_map)

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
    
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def infer_anchor_center(model, criterion, data_loader, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    
    
    predictions = dict()
    predictions["categories"] = [{'name': 'fg', 'id': 1}]
    predictions["images"] = list()
    predictions["annotations"] = list()
    predictions["sampled_points"] = list()

    anno_id = 1

    # output_dir_centerness_dir = os.path.join(output_dir, "centerness_map")
    output_dir_centerness_dir = os.path.join("./outputs/fscd_147_anchor_centerness/", "centerness_map")
    os.makedirs(output_dir_centerness_dir, exist_ok=True)

    for ret in metric_logger.log_every(data_loader, 100, header):
        image = ret['image'].to(device)
        orig_size = ret['orig_size']

        if 'sampled_points' in ret.keys():
           sampled_points = ret['sampled_points'].squeeze(0).numpy()
        else: sampled_points = None
        targets = [{
            "boxes": ret['boxes'].squeeze(0).to(device),
            "labels": ret['labels'].squeeze(0).to(device), 
            "centerness_map": ret["centerness_map"].squeeze(0).to(device), 

        }]

        ori_shape = ret["orig_size"].detach().cpu().numpy()
        ori_h, ori_w = ori_shape[0]

        all_outputs = model(image, orig_size=orig_size)
        ori_outputs, centerness_map, ref_points = all_outputs[0], all_outputs[1], all_outputs[2]
        outputs = (ori_outputs, centerness_map)

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

        out_logits, out_bbox = ori_outputs['pred_logits'], ori_outputs['pred_boxes']

        prob = out_logits.sigmoid()
        object_prob = prob[...,0]
        obj_pos = torch.where(object_prob >= 0.5)
            
        pred_scores = object_prob[obj_pos[0],obj_pos[1]]
        pred_boxes = out_bbox[obj_pos[0],obj_pos[1]]
        ref_points = ref_points[obj_pos[0],obj_pos[1]]
        pred_points = ref_points.detach().cpu().numpy()
        pred_points[...,0] *= ori_w
        pred_points[...,1] *= ori_h

        pred_boxes = pred_boxes.detach().cpu().numpy()
        pred_boxes[...,0] *= ori_w; pred_boxes[...,1] *= ori_h
        pred_boxes[...,2] *= ori_w; pred_boxes[...,3] *= ori_h
        
        image_id = ret["image_id"].item()
        ori_h, ori_w = ret["orig_size"].numpy()[0]

        pred_centerness_map_path = os.path.join(output_dir_centerness_dir, str(image_id) + ".jpg")
        centerness_map = nn.Sigmoid()(centerness_map)
        centerness_map = torch.squeeze(centerness_map)
        centerness_map_array = centerness_map.detach().cpu().numpy()
        # pred_heatmap = None
        # pred_heatmap = cv2.normalize(centerness_map_array, pred_heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # pred_heatmap = cv2.applyColorMap(pred_heatmap, cv2.COLORMAP_JET)
        pred_heatmap = centerness_map_array * 255
        pred_heatmap = pred_heatmap.astype(int)
        
        cv2.imwrite(pred_centerness_map_path, pred_heatmap)

        img_info = {
            "id": image_id, 
            "height": int(ori_h),
            "width": int(ori_w),
            "file_name": "None",
            # "points": sampled_points, 
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

    with open(os.path.join(output_dir, "predictions.json"), "w") as handle:
        json.dump(predictions, handle)



    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model,criterion,data_loader, device):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    for ret in metric_logger.log_every(data_loader, 100, header):
        image = ret['image'].to(device)
        if 'sampled_points' in ret.keys():
           sampled_points = ret['sampled_points'].squeeze(0).numpy()
        else: 
            sampled_points = None
        orig_size = ret["orig_size"]
        all_outputs = model(image)
        outputs = all_outputs[0]; ref_points = all_outputs[1]
        targets = [{
            "boxes": ret['boxes'].squeeze(0).to(device),
            "labels": ret['labels'].squeeze(0).to(device), 
        }]
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

    
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def infer_sample(model, criterion, data_loader, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    predictions = dict()
    predictions["categories"] = [{'name': 'fg', 'id': 1}]
    predictions["images"] = list()
    predictions["annotations"] = list()
    predictions["sampled_points"] = list()

    anno_id = 1        
    
    for ret in metric_logger.log_every(data_loader, 100, header):
        image = ret['image'].to(device)
        if 'sampled_points' in ret.keys():
           sampled_points = ret['sampled_points'].squeeze(0).numpy()
        else: 
            sampled_points = None
        all_outputs = model(image, sampled_points)
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
        obj_pos = torch.where(object_prob >= 0.5)
            
        pred_scores = object_prob[obj_pos[0],obj_pos[1]]
        pred_boxes = out_bbox[obj_pos[0],obj_pos[1]]
        ref_points = ref_points[obj_pos[0],obj_pos[1]]
        pred_points = ref_points.detach().cpu().numpy()
        pred_points[...,0] *= ori_w
        pred_points[...,1] *= ori_h

        pred_boxes = pred_boxes.detach().cpu().numpy()
        pred_boxes[...,0] *= ori_w; pred_boxes[...,1] *= ori_h
        pred_boxes[...,2] *= ori_w; pred_boxes[...,3] *= ori_h

        
        image_id = ret["image_id"].item()
        ori_h, ori_w = ret["orig_size"].numpy()[0]

        sampled_points[0, :] *= ori_w
        sampled_points[1, :] *= ori_h
        sampled_points = sampled_points.astype(int)
        sampled_points = sampled_points.tolist()
        img_info = {
            "id": image_id, 
            "height": int(ori_h),
            "width": int(ori_w),
            "file_name": "None",
            "points": sampled_points, 
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

    with open(os.path.join(output_dir, "predictions.json"), "w") as handle:
        json.dump(predictions, handle)

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
