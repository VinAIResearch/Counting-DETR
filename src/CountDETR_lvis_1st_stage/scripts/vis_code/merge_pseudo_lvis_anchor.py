import os
import os.path as osp
import json
from pycocotools.coco import COCO
import numpy as np
import cv2
import random as rd

folder = "./FSCD_LVIS/Pseudo_LVIS/"
json_path = osp.join(folder, "pseudo_bbox_train_p1.json")
contents = json.load(open(json_path, "r"))

instances_train = json.load(open(osp.join("FSCD_LVIS", "instances_train.json"), "r"))
train_images = instances_train["images"]
train_map_id_2_image = dict()
train_map_image_2_id = dict()
for train_image in train_images:
    img_id = train_image["id"]
    img_name = train_image["file_name"]
    train_map_id_2_image[img_id] = img_name
    train_map_image_2_id[img_name] = img_id

instances_val = json.load(open(osp.join("FSCD_LVIS", "instances_val.json"), "r"))
val_images = instances_val["images"]
val_map_id_2_image = dict()
val_map_image_2_id = dict()
for val_image in val_images:
    img_id = val_image["id"]
    img_name = val_image["file_name"]
    val_map_id_2_image[img_id] = img_name
    val_map_image_2_id[img_name] = img_id


pseudo_train = dict()
pseudo_train["licenses"] = contents["licenses"]
pseudo_train["info"] = contents["info"]
pseudo_train["categories"] = contents["categories"]
pseudo_train["images"] = []
pseudo_train["annotations"] = []

pseudo_val = dict()
pseudo_val["licenses"] = contents["licenses"]
pseudo_val["info"] = contents["info"]
pseudo_val["categories"] = contents["categories"]
pseudo_val["images"] = []
pseudo_val["annotations"] = []

all_img_ids = []
train_image_id = 1
train_anno_id = 1
val_image_id = 1
val_anno_id = 1

for i in range(1, 5):
    json_path = osp.join(folder, "pseudo_bbox_train_p"+str(i)+".json")
    this_contents = json.load(open(json_path, "r"))
    coco_api = COCO(json_path)
    this_images = this_contents["images"]

    for this_image in this_images:
        img_id = this_image["id"]
        img_name = this_image["file_name"]
        new_img_info = this_image.copy()
        if img_name in train_map_image_2_id.keys():
            anno_ids = coco_api.getAnnIds([img_id])
            annos = coco_api.loadAnns(anno_ids)
            new_img_info["id"] = train_image_id
            for anno in annos:
                new_anno = anno.copy()
                new_anno["image_id"] = train_image_id
                new_anno["id"] = train_anno_id
                pseudo_train["annotations"].append(new_anno)
                train_anno_id += 1
            pseudo_train["images"].append(new_img_info)
            train_image_id += 1
        else:
            assert img_name in val_map_image_2_id.keys()
            anno_ids = coco_api.getAnnIds([img_id])
            annos = coco_api.loadAnns(anno_ids)
            new_img_info["id"] = val_image_id
            for anno in annos:
                new_anno = anno.copy()
                new_anno["image_id"] = val_image_id
                new_anno["id"] = val_anno_id
                pseudo_val["annotations"].append(new_anno)
                val_anno_id += 1
            pseudo_val["images"].append(new_img_info)
            val_image_id += 1

with open("./FSCD_LVIS/pseudo_lvis_train.json", "w") as handle:
    json.dump(pseudo_train, handle)

with open("./FSCD_LVIS/pseudo_lvis_val.json", "w") as handle:
    json.dump(pseudo_val, handle)

