import json
import os
import random as rd

import cv2
import numpy as np
from pycocotools.coco import COCO


for split in ["train", "val"]:
    JSON_PATH = "./FSCD_LVIS/instances_" + split + ".json"
    with open(JSON_PATH, "r") as handle:
        contents = json.load(handle)
    coco_api = COCO(JSON_PATH)
    box_point_dict = dict()
    box_point_dict["images"] = contents["images"]
    box_point_dict["categories"] = contents["categories"]
    box_point_dict["annotations"] = list()

    all_imaged_ids = coco_api.getImgIds()
    for image_id in all_imaged_ids:
        anno_ids = coco_api.getAnnIds([image_id])
        annos = coco_api.loadAnns(anno_ids)
        all_points = []
        all_boxes = []
        for anno in annos:
            points = []
            segmentations = anno["segmentation"]
            num_points = int(len(segmentations[0]) / 2)
            for i in range(num_points):
                x = int(segmentations[0][2 * i])
                y = int(segmentations[0][2 * i + 1])
                points.append([x, y])
            current_contour = np.array(points)
            current_contour = np.reshape(current_contour, (num_points, 1, 2))
            try:
                M = cv2.moments(current_contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            except Exception as e:
                xs, ys = [], []
                for i in range(num_points):
                    x = segmentations[0][2 * i]
                    y = segmentations[0][2 * i + 1]
                    xs.append(x)
                    ys.append(y)
                cX = np.mean(np.array(xs))
                cY = np.mean(np.array(ys))
            center_point = [cX, cY]
            new_anno = anno.copy()
            new_anno["center"] = center_point
            box = anno["bbox"]
            all_points.append(center_point)
            all_boxes.append(box)
        import pdb

        pdb.set_trace()
        exemplar_boxes = rd.sample(all_boxes, 5)
        img_anno = {"image_id": image_id, "points": all_points, "exemplars_boxes": exemplar_boxes}
        box_point_dict["annotations"].append(img_anno)
    with open("./FSCD_LVIS/counting_points_boxes_" + split + ".json", "w") as handle:
        json.dump(box_point_dict, handle)
