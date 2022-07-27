import json
import os
from os.path import join

import cv2
import numpy as np
from numpy.linalg import norm
from pycocotools.coco import COCO
from scipy.ndimage.filters import gaussian_filter


data_folder = "./FSCD_LVIS/"
for split in ["train", "val"]:
    json_path = join(data_folder, "split_point_and_box_" + split + ".json")
    contents = json.load(open(json_path, "r"))
    annos = contents["annotations"]
    images = contents["images"]
    map_id_2_name = dict()
    map_name_2_id = dict()
    for img_info, anno in zip(images, annos):
        assert img_info["id"] == anno["image_id"]
        mask_path = join(data_folder, "new_masks", split, str(img_info["id"]) + ".npy")
        points = anno["points"]
        points = np.array(points)
        distances = []
        for i, first_point in enumerate(points):
            min_dist = 100.0
            for j, second_point in enumerate(points):
                if j == i:
                    continue
                dist = norm(first_point - second_point)
                if dist < min_dist:
                    min_dist = dist
            distances.append(min_dist)
        std = np.mean(distances) / 4
        final_mask = np.zeros((img_info["height"], img_info["width"]))

        for i, point in enumerate(points):
            mask = np.zeros((img_info["height"], img_info["width"]))
            x, y = point
            mask[int(y), int(x)] = 1
            mask = gaussian_filter(mask, sigma=std)
            final_mask += mask
            del mask
        np.save(mask_path, final_mask)
        print(mask_path)
