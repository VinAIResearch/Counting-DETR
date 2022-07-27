import json
import os
import random as rd
from os.path import join

from pycocotools.coco import COCO


json_path = "./FSCD_LVIS/point_and_box_train.json"
point_and_box_annotations = json.load(open(json_path, "r"))

json_path = "./FSCD_LVIS/instances_train.json"
annotations = json.load(open(json_path, "r"))

coco_api = COCO(json_path)
all_img_ids = coco_api.getImgIds()

im_23 = []
im_34 = []
im_45 = []
im_56 = []
im_6x = []
for img_id in all_img_ids:
    anno_ids = coco_api.getAnnIds([img_id])
    annos = coco_api.loadAnns(anno_ids)
    if len(anno_ids) <= 30:
        im_23.append(img_id)
    elif len(anno_ids) <= 40 and len(anno_ids) > 30:
        im_34.append(img_id)
    elif len(anno_ids) <= 50 and len(anno_ids) > 40:
        im_45.append(img_id)
    elif len(anno_ids) <= 60 and len(anno_ids) > 50:
        im_56.append(img_id)
    else:
        im_6x.append(img_id)

rd.shuffle(im_23)
rd.shuffle(im_34)
rd.shuffle(im_45)
rd.shuffle(im_56)
rd.shuffle(im_6x)

train_id = []
val_id = []
num_train_23 = int(0.8 * len(im_23))
train_id.extend(im_23[:num_train_23])
val_id.extend(im_23[num_train_23:])

num_train_34 = int(0.8 * len(im_34))
train_id.extend(im_34[:num_train_34])
val_id.extend(im_34[num_train_34:])

num_train_45 = int(0.8 * len(im_45))
train_id.extend(im_45[:num_train_45])
val_id.extend(im_45[num_train_45:])

num_train_56 = int(0.8 * len(im_56))
train_id.extend(im_56[:num_train_56])
val_id.extend(im_56[num_train_56:])

num_train_6x = int(0.8 * len(im_6x))
train_id.extend(im_6x[:num_train_6x])
val_id.extend(im_6x[num_train_6x:])

train_dict = dict()
train_dict["categories"] = annotations["categories"]
train_dict["annotations"] = []
train_dict["images"] = []
original_images = annotations["images"]

point_box_images = point_and_box_annotations["images"]
point_box_annos = point_and_box_annotations["annotations"]

train_point_box = dict()
train_point_box["images"] = []
train_point_box["annotations"] = []

new_img_id = 1
new_anno_id = 1
try:
    for old_img_id in train_id:
        old_img_info = original_images[old_img_id - 1]
        new_image_info = {
            "id": new_img_id,
            "width": old_img_info["width"],
            "height": old_img_info["height"],
            "file_name": old_img_info["file_name"],
        }
        anno_ids = coco_api.getAnnIds([old_img_id])
        old_annos = coco_api.loadAnns(anno_ids)
        for old_anno in old_annos:
            new_anno = old_anno.copy()
            new_anno["image_id"] = new_img_id
            new_anno["id"] = new_anno_id
            train_dict["annotations"].append(new_anno)
            new_anno_id += 1
        train_dict["images"].append(new_image_info)
        new_train_point_box_images = point_box_images[old_img_id - 1].copy()
        new_train_point_box_annos = point_box_annos[old_img_id - 1].copy()
        new_train_point_box_images["id"] = new_img_id
        new_train_point_box_annos["image_id"] = new_img_id
        train_point_box["images"].append(new_train_point_box_images)
        train_point_box["annotations"].append(new_train_point_box_annos)
        new_img_id += 1
except Exception as e:
    print(e)
    import sys

    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)
    import pdb

    pdb.set_trace()
print(len(train_dict["images"]), len(train_dict["annotations"]))

new_json_path = "./FSCD_LVIS/split_instances_train.json"
with open(new_json_path, "w") as handle:
    json.dump(train_dict, handle)


new_json_path = "./FSCD_LVIS/split_point_and_box_train.json"
with open(new_json_path, "w") as handle:
    json.dump(train_point_box, handle)


val_dict = dict()
val_dict["categories"] = annotations["categories"]
val_dict["annotations"] = []
val_dict["images"] = []
original_images = annotations["images"]

val_point_box = dict()
val_point_box["images"] = []
val_point_box["annotations"] = []

new_img_id = 1
new_anno_id = 1
try:
    for old_img_id in val_id:
        old_img_info = original_images[old_img_id - 1]
        new_image_info = {
            "id": new_img_id,
            "width": old_img_info["width"],
            "height": old_img_info["height"],
            "file_name": old_img_info["file_name"],
        }
        anno_ids = coco_api.getAnnIds([old_img_id])
        old_annos = coco_api.loadAnns(anno_ids)
        for old_anno in old_annos:
            new_anno = old_anno.copy()
            new_anno["image_id"] = new_img_id
            new_anno["id"] = new_anno_id
            val_dict["annotations"].append(new_anno)
            new_anno_id += 1
        val_dict["images"].append(new_image_info)

        new_val_point_box_images = point_box_images[old_img_id - 1].copy()
        new_val_point_box_annos = point_box_annos[old_img_id - 1].copy()
        new_val_point_box_images["id"] = new_img_id
        new_val_point_box_annos["image_id"] = new_img_id
        val_point_box["images"].append(new_val_point_box_images)
        val_point_box["annotations"].append(new_val_point_box_annos)
        new_img_id += 1

except Exception as e:
    print(e)
    import sys

    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)
    import pdb

    pdb.set_trace()
print(len(val_dict["images"]), len(val_dict["annotations"]))

new_json_path = "./FSCD_LVIS/split_instances_val.json"
with open(new_json_path, "w") as handle:
    json.dump(val_dict, handle)

new_json_path = "./FSCD_LVIS/split_point_and_box_val.json"
with open(new_json_path, "w") as handle:
    json.dump(val_point_box, handle)
