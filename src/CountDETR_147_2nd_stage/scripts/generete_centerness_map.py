import os
import numpy as np
import argparse
import json
import math
import cv2
from pycocotools.coco import COCO


def load_json(json_file):
    with open(json_file,'r') as f:
        data = json.load(f)
    return data

def get_centerness_map(coco_anno, image_id):
    return 

def convert_to_centerness_map(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    pseudo_train = os.path.join(input_folder,'pseudo_train_anchor_detr.json')
    pseudo_val = os.path.join(input_folder,'pseudo_val_anchor_detr.json')
    image_folder = os.path.join(input_folder, "images_384_VarV2")
    pseudo_label_train = COCO(pseudo_train)
    train_image_ids = pseudo_label_train.getImgIds()

    for image_id in train_image_ids:
        anno_ids = pseudo_label_train.getAnnIds(image_id)
        annos = pseudo_label_train.loadAnns(anno_ids)
        image_info = pseudo_label_train.loadImgs(image_id)
        file_name = image_info[0]["file_name"]
        file_id = file_name[:-4]
        image_path = os.path.join(image_folder, file_name)
        img_height, img_width = image_info[0]["height"], image_info[0]["width"]
        centerness_map = np.zeros((img_height, img_width), dtype=np.float32)
        boxes = [anno["bbox"] for anno in annos]
        boxes = np.array(boxes)
        boxes[:, 0] = np.clip(boxes[:, 0], 0, img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, img_height)
        box_centers = [((x0 + x1)/2, (y0 + y1)/2) for x0, y0, x1, y1 in boxes]
        print(file_id, img_height, img_width, len(box_centers), end=" ")
        output_path = os.path.join(output_folder, file_id + ".npy")
        for bbox, box_center in zip(boxes, box_centers):
            x0, y0, x1, y1 = bbox
            x_cen, y_cen = box_center
            for i in range(x0, x1):
                for j in range(y0, y1):     
                    l, r, t, b = i - x0, x1 - i, j - y0, y1 - j 
                    min_lr = min(l, r); max_lr = max(l, r)
                    min_tb = min(t, b); max_tb = max(t, b)
                    centerness = math.sqrt((min_lr*min_tb)/ (max_lr*max_tb + 1e-6))
                    centerness_map[j][i] += centerness
        output_path = os.path.join(output_folder, file_id + ".npy")
        with open(output_path, "wb") as handle:
            np.save(handle, centerness_map)
            print("Save {}".format(output_path))
        image = cv2.imread(image_path)
        gt_heatmap = None
        gt_heatmap = cv2.normalize(centerness_map, gt_heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        gt_heatmap = cv2.applyColorMap(gt_heatmap, cv2.COLORMAP_JET)
        merged_image = cv2.hconcat([image, gt_heatmap])
        vis_output_path = os.path.join(output_folder, file_id + "_vis.jpg")
        cv2.imwrite(vis_output_path, merged_image)
        print(vis_output_path)

    pseudo_label_val = COCO(pseudo_val)
    val_image_ids = pseudo_label_val.getImgIds()
    for image_id in val_image_ids:
        anno_ids = pseudo_label_val.getAnnIds(image_id)
        annos = pseudo_label_val.loadAnns(anno_ids)
        image_info = pseudo_label_val.loadImgs(image_id)
        file_name = image_info[0]["file_name"]
        image_path = os.path.join(image_folder, file_name)
        file_id = file_name[:-4]
        img_height, img_width = image_info[0]["height"], image_info[0]["width"]
        centerness_map = np.zeros((img_height, img_width), dtype=np.float32)
        boxes = [anno["bbox"] for anno in annos]
        boxes = np.array(boxes)
        boxes[:, 0] = np.clip(boxes[:, 0], 0, img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, img_height)
        box_centers = [((x0 + x1)/2, (y0 + y1)/2) for x0, y0, x1, y1 in boxes]
        print(file_id, img_height, img_width, end=" ")

        for bbox, box_center in zip(boxes, box_centers):
            x0, y0, x1, y1 = bbox
            x_cen, y_cen = box_center
            for i in range(x0, x1):
                for j in range(y0, y1):     
                    l, r, t, b = i - x0, x1 - i, j - y0, y1 - j 
                    min_lr = min(l, r); max_lr = max(l, r)
                    min_tb = min(t, b); max_tb = max(t, b)
                    centerness = math.sqrt((min_lr*min_tb)/ (max_lr*max_tb + 1e-6))
                    centerness_map[j][i] += centerness
        output_path = os.path.join(output_folder, file_id + ".npy")
        with open(output_path, "wb") as handle:
            np.save(handle, centerness_map)
            print("Save {}".format(output_path))
        image = cv2.imread(image_path)
        gt_heatmap = None
        gt_heatmap = cv2.normalize(centerness_map, gt_heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        gt_heatmap = cv2.applyColorMap(gt_heatmap, cv2.COLORMAP_JET)
        merged_image = cv2.hconcat([image, gt_heatmap])
        vis_output_path = os.path.join(output_folder, file_id + "_vis.jpg")
        cv2.imwrite(vis_output_path, merged_image)
        print(vis_output_path)

    return 

def convert_to_centerness_map_test(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    test_json = os.path.join(input_folder,'instances_test.json')
    label_test = COCO(test_json)
    image_ids = label_test.getImgIds()
    image_folder = os.path.join(input_folder, "images_384_VarV2")
    for image_id in image_ids:
        anno_ids = label_test.getAnnIds(image_id)
        annos = label_test.loadAnns(anno_ids)
        image_info = label_test.loadImgs(image_id)
        file_name = image_info[0]["file_name"]
        image_path = os.path.join(image_folder, file_name)
        file_id = file_name[:-4]
        img_height, img_width = image_info[0]["height"], image_info[0]["width"]
        centerness_map = np.zeros((img_height, img_width), dtype=np.float32)
        boxes = [anno["bbox"] for anno in annos]
        boxes = np.array(boxes)
        boxes[:, 0] = np.clip(boxes[:, 0], 0, img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, img_height)
        box_centers = [((x0 + x1)/2, (y0 + y1)/2) for x0, y0, x1, y1 in boxes]
        print(file_id, img_height, img_width, end=" ")
        output_path = os.path.join(output_folder, file_id + ".npy")
        for bbox, box_center in zip(boxes, box_centers):
            x0, y0, x1, y1 = bbox
            x_cen, y_cen = box_center
            for i in range(x0, x1):
                for j in range(y0, y1):     
                    l, r, t, b = i - x0, x1 - i, j - y0, y1 - j 
                    min_lr = min(l, r); max_lr = max(l, r)
                    min_tb = min(t, b); max_tb = max(t, b)
                    centerness = math.sqrt((min_lr*min_tb)/ (max_lr*max_tb + 1e-6))
                    centerness_map[j][i] += centerness
        output_path = os.path.join(output_folder, file_id + ".npy")
        with open(output_path, "wb") as handle:
            np.save(handle, centerness_map)
            print("Save {}".format(output_path))
        image = cv2.imread(image_path)
        gt_heatmap = None
        gt_heatmap = cv2.normalize(centerness_map, gt_heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        gt_heatmap = cv2.applyColorMap(gt_heatmap, cv2.COLORMAP_JET)
        merged_image = cv2.hconcat([image, gt_heatmap])
        vis_output_path = os.path.join(output_folder, file_id + "_vis.jpg")
        cv2.imwrite(vis_output_path, merged_image)
        print(vis_output_path)
    return 

def get_args_parser():
    parser = argparse.ArgumentParser('Generate centerness map', add_help=False)
    parser.add_argument('--input_folder', default="./FSC147", type=str)
    parser.add_argument('--output_folder', default="./FSC147/centerness_map", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args_parser()
    convert_to_centerness_map(args.input_folder, args.output_folder)
    convert_to_centerness_map_test(args.input_folder, args.output_folder)
