import json
import os
import random

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch import dtype
from torch.utils.data import Dataset
from torchvision import transforms as transforms


class FSCD147_Exemplars(Dataset):
    def __init__(self, args, split="train", mode=None):
        print("This data is fscd 147, with few exmplar boxes and points, split: {}".format(split))
        data_path = args.data_path
        self.anno_file = os.path.join(data_path, "annotation_FSC147_384.json")
        self.data_split_file = os.path.join(data_path, "Train_Test_Val_FSC_147.json")
        self.im_dir = os.path.join(data_path, "images_384_VarV2")
        self.scale_factor = args.scale_factor

        self.annotations = self.load_json(self.anno_file)
        self.data_split = self.load_json(self.data_split_file)[split]

        self.mode = mode
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    def load_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)

        return data

    def __len__(self):
        return len(self.data_split)

    def __getitem__(self, idx):
        im_id = self.data_split[idx]
        anno = self.annotations[im_id]
        bboxes = anno["box_examples_coordinates"]

        box_center = list()
        whs = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            box_center.append([(x1 + x2) / 2, (y1 + y2) / 2])
            whs.append([x2 - x1, y2 - y1])

        box_center = np.array(box_center, dtype=np.float32)  # [N_boxes,2]
        whs = np.array(whs, dtype=np.float32)  # [N_boxes,2]
        image = Image.open("{}/{}".format(self.im_dir, im_id))
        img_w, img_h = image.size
        orig_size = np.array([img_w, img_h])

        resize_w = 32 * int(img_w / 32)
        resize_h = 32 * int(img_h / 32)
        image = image.resize((resize_w, resize_h), Image.BILINEAR)
        image = self.transform(image)

        img_res = np.array([img_w, img_h], dtype=np.float32)

        scaled_whs = whs / img_res[None, :]

        points = box_center / img_res[None, :]
        labels = np.zeros(points.shape[0], dtype=np.int64)

        sample = {
            "image": image,
            "points": points,
            "whs": scaled_whs,
            "labels": labels,
            "orig_size": orig_size,
        }
        return sample


class FSCD147_Points(Dataset):
    def __init__(self, args, split="train", mode=None):
        print("This data is fscd 147, with points only, to generate pseudo label, split: {}".format(split))
        data_path = args.data_path
        self.anno_file = os.path.join(data_path, "annotation_FSC147_384.json")
        self.data_split_file = os.path.join(data_path, "Train_Test_Val_FSC_147.json")
        self.im_dir = os.path.join(data_path, "images_384_VarV2")
        self.scale_factor = args.scale_factor

        self.annotations = self.load_json(self.anno_file)
        self.data_split = self.load_json(self.data_split_file)[split]

        self.mode = mode
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    def load_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)

        return data

    def __len__(self):
        return len(self.data_split)

    def __getitem__(self, idx):
        im_id = self.data_split[idx]
        anno = self.annotations[im_id]
        bboxes = anno["box_examples_coordinates"]
        all_points = np.array(anno["points"], dtype=np.float32)
        box_center = list()
        whs = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            box_center.append([(x1 + x2) / 2, (y1 + y2) / 2])
            whs.append([x2 - x1, y2 - y1])

        box_center = np.array(box_center, dtype=np.float32)  # [N_boxes,2]
        whs = np.array(whs, dtype=np.float32)  # [N_boxes,2]
        image = Image.open("{}/{}".format(self.im_dir, im_id))
        img_w, img_h = image.size
        orig_size = np.array([img_w, img_h])
        img_res = np.array([img_w, img_h], dtype=np.float32)

        resize_w = self.scale_factor * int(img_w / self.scale_factor)
        resize_h = self.scale_factor * int(img_h / self.scale_factor)
        image = image.resize((resize_w, resize_h), Image.BILINEAR)
        img_w, img_h = image.size

        image = self.transform(image)

        anchor_points = box_center / img_res[None, :]
        all_points = all_points / img_res[None, :]
        labels = np.zeros(all_points.shape[0], dtype=np.int64)

        ret = {
            "im_id": int(im_id[:-4]),
            "image": image,
            "points": all_points,
            "labels": labels,
            "anchor_points": anchor_points,
            "orig_size": orig_size,
        }
        return ret


class FSCD147_Test(Dataset):
    def __init__(self, args, split="test", mode=None):
        print("This data is fscd 147 test set, split: {}".format(split))
        data_path = args.data_path
        self.anno_file = os.path.join(data_path, "annotation_FSC147_384.json")
        self.data_split_file = os.path.join(data_path, "Train_Test_Val_FSC_147.json")
        self.im_dir = os.path.join(data_path, "images_384_VarV2")
        self.test_file = os.path.join(data_path, "instances_test.json")
        self.scale_factor = args.scale_factor

        self.annotations = self.load_json(self.anno_file)
        self.data_split = self.load_json(self.data_split_file)[split]
        self.label_test = COCO(self.test_file)
        self.img_name_to_ori_id = self.map_img_name_to_ori_id()
        self.mode = mode
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        print("This data contains: {} images".format(len(self.data_split)))

    def load_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)

        return data

    def map_img_name_to_ori_id(self,):
        all_coco_imgs = self.label_test.imgs
        map_name_2_id = dict()
        for k, v in all_coco_imgs.items():
            img_id = v["id"]
            img_name = v["file_name"]
            map_name_2_id[img_name] = img_id
        return map_name_2_id

    def __len__(self):
        return len(self.data_split)

    def __getitem__(self, idx):
        img_name = self.data_split[idx]
        coco_im_id = self.img_name_to_ori_id[img_name]
        anno_ids = self.label_test.getAnnIds([coco_im_id])
        annos = self.label_test.loadAnns(anno_ids)

        box_centers = list()
        whs = list()
        for anno in annos:
            bbox = anno["bbox"]
            x1, y1, w, h = bbox
            box_centers.append([x1 + w / 2, y1 + h / 2])
            whs.append([w, h])

        ori_exemplar_boxes = self.annotations[img_name]["box_examples_coordinates"]
        exemplar_boxes = list()
        for exemplar_box in ori_exemplar_boxes:
            y1 = exemplar_box[0][1]
            x1 = exemplar_box[0][0]
            x2 = exemplar_box[2][0]
            y2 = exemplar_box[2][1]
            exemplar_boxes.append([x1, y1, x2, y2])
        exemplar_boxes = np.array(exemplar_boxes, dtype=np.float32)

        box_centers = np.array(box_centers, dtype=np.float32)  # [N_boxes,2]
        whs = np.array(whs, dtype=np.float32)  # [N_boxes,2]
        boxes = np.concatenate((box_centers, whs), axis=1)

        image = Image.open("{}/{}".format(self.im_dir, img_name))
        img_w, img_h = image.size
        orig_size = np.array([img_w, img_h])
        img_res = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
        scaled_boxes = boxes / img_res[None, :]
        scaled_exemplar_boxes = exemplar_boxes / img_res[None, :]

        img_res = np.array([img_w, img_h], dtype=np.float32)
        scaled_points = box_centers / img_res[None, :]

        resize_w = self.scale_factor * int(img_w / self.scale_factor)
        resize_h = self.scale_factor * int(img_h / self.scale_factor)
        image = image.resize((resize_w, resize_h), Image.BILINEAR)
        image = self.transform(image)
        labels = np.zeros(scaled_points.shape[0], dtype=np.int64)
        ret = {
            "im_id": int(img_name[:-4]),
            "image_id": coco_im_id,
            "image": image,
            "points": scaled_points,
            "boxes": scaled_boxes,
            "orig_size": orig_size,
            "exemplar_boxes": scaled_exemplar_boxes,
            "labels": labels,
            "ori_points": box_centers,
        }
        return ret


def build_fscd_147(args, image_set):
    return FSCD147_Exemplars(args, image_set)


def build_fscd_147_points(args, image_set):
    return FSCD147_Points(args, image_set)


def build_fscd_test(args, image_set):
    return FSCD147_Test(args, image_set)
