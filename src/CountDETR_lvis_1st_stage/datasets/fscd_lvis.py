import json
import os
import random

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms as transforms


class FSCD_LVIS_Exemplars(Dataset):
    def __init__(self, args, split="train", mode=None):
        print("This data is fscd LVIS, with few exmplar boxes and points, split: {}".format(split), end="  ")
        data_path = args.data_path
        pseudo_label_file = "instances_" + split + ".json"
        self.coco = COCO(os.path.join(data_path, "annotations", pseudo_label_file))
        self.image_ids = self.coco.getImgIds()

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        self.img_path = os.path.join(data_path, "images", "all_images")
        self.count_anno_file = os.path.join(data_path, "annotations", "count_" + split + ".json")
        self.count_anno = self.load_json(self.count_anno_file)

        self.scale_factor = args.scale_factor

        self.mode = mode
        print("with number of images: ", self.__len__())

    def load_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)

        return data

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_file = img_info["file_name"]

        # ann_ids = self.coco.getAnnIds([img_id])
        # anns = self.coco.loadAnns(ids=ann_ids)
        # bboxes = np.array([instance['bbox'] for instance in anns],dtype=np.float32)

        ex_bboxes = self.count_anno["annotations"][idx]["boxes"]

        points = list()
        whs = list()
        for bbox in ex_bboxes[:3]:
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            x_cen, y_cen = (x1 + x2) / 2, (y1 + y2) / 2
            points.append([x_cen, y_cen])
            whs.append([w, h])

        points = np.array(points, dtype=np.float32)  # [N_boxes,2]

        whs = np.array(whs, dtype=np.float32)  # [N_boxes,2]
        img = Image.open(os.path.join(self.img_path, img_file))
        img = img.convert("RGB")

        img_w, img_h = img.size
        orig_size = np.array([img_w, img_h])

        resize_w = 32 * int(img_w / 32)
        resize_h = 32 * int(img_h / 32)
        image = img.resize((resize_w, resize_h), Image.BILINEAR)
        image = self.transform(image)

        img_res = np.array([img_w, img_h], dtype=np.float32)

        scaled_whs = whs / img_res[None, :]

        points = points / img_res[None, :]
        labels = np.zeros(points.shape[0], dtype=np.int64)

        img_res = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)

        # img_name = img_file.split(".")[0]
        img_name = img_file
        sample = {
            "img_name": img_name,
            "image": image,
            "points": points,
            "whs": scaled_whs,
            "labels": labels,
            "orig_size": orig_size,
        }
        return sample


class FSCD_LVIS_Points(Dataset):
    def __init__(self, args, split="train", mode=None):
        print("This data is fscd LVIS, with few exmplar boxes and points, split: {}".format(split), end="   ")
        data_path = args.data_path
        pseudo_label_file = "instances_" + split + ".json"
        self.coco = COCO(os.path.join(data_path, "annotations", pseudo_label_file))
        self.image_ids = self.coco.getImgIds()

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        self.img_path = os.path.join(data_path, "images", "all_images")
        self.count_anno_file = os.path.join(data_path, "annotations", "count_" + split + ".json")
        self.count_anno = self.load_json(self.count_anno_file)

        self.scale_factor = args.scale_factor

        self.mode = mode
        print("with number of images: ", self.__len__())

    def load_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)

        return data

    def __len__(self):
        return len(self.count_anno["images"])

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_file = img_info["file_name"]

        ann_ids = self.coco.getAnnIds([img_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        ex_bboxes = self.count_anno["annotations"][idx]["boxes"]
        all_points = np.array(self.count_anno["annotations"][idx]["points"], dtype=np.float32)
        assert self.count_anno["images"][idx]["file_name"] == img_file

        points = list()
        whs = list()
        for bbox in ex_bboxes[:3]:
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            x_cen, y_cen = (x1 + x2) / 2, (y1 + y2) / 2
            points.append([x_cen, y_cen])
            whs.append([w, h])

        points = np.array(points, dtype=np.float32)  # [N_boxes,2]
        whs = np.array(whs, dtype=np.float32)  # [N_boxes,2]
        img = Image.open(os.path.join(self.img_path, img_file))
        img = img.convert("RGB")

        img_w, img_h = img.size
        orig_size = np.array([img_w, img_h])

        resize_w = 32 * int(img_w / 32)
        resize_h = 32 * int(img_h / 32)
        image = img.resize((resize_w, resize_h), Image.BILINEAR)
        image = self.transform(image)

        img_res = np.array([img_w, img_h], dtype=np.float32)

        scaled_whs = whs / img_res[None, :]

        anchor_points = points / img_res[None, :]
        all_points = all_points / img_res[None, :]
        labels = np.zeros(points.shape[0], dtype=np.int64)

        img_res = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)

        # img_name = int(img_file.split(".")[0])
        img_name = img_file

        sample = {
            "img_name": img_name,
            "image": image,
            "anchor_points": anchor_points,
            "whs": scaled_whs,
            "labels": labels,
            "orig_size": orig_size,
            "points": all_points,
        }
        return sample


class FSCD_LVIS_Test(Dataset):
    def __init__(self, args, split="test", mode=None):
        print("This data is fscd LVIS, with few exmplar boxes and points, split: {}".format(split), end="   ")
        data_path = args.data_path
        pseudo_label_file = "instances_" + split + ".json"
        self.coco = COCO(os.path.join(data_path, "annotations", pseudo_label_file))
        self.image_ids = self.coco.getImgIds()

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        self.img_path = os.path.join(data_path, "images", "all_images")
        self.count_anno_file = os.path.join(data_path, "annotations", "count_" + split + ".json")
        self.count_anno = self.load_json(self.count_anno_file)

        self.scale_factor = args.scale_factor

        self.mode = mode
        print("with number of images: ", self.__len__())

    def load_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)

        return data

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_file = img_info["file_name"]

        ann_ids = self.coco.getAnnIds([img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        bboxes = np.array([instance["bbox"] for instance in anns], dtype=np.float32)
        ex_bboxes = self.count_anno["annotations"][idx]["boxes"]
        all_points = np.array(self.count_anno["annotations"][idx]["points"], dtype=np.float32)

        assert self.count_anno["images"][idx]["file_name"] == img_file

        points = list()
        whs = list()
        for bbox in ex_bboxes[:3]:
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            x_cen, y_cen = (x1 + x2) / 2, (y1 + y2) / 2
            points.append([x_cen, y_cen])
            whs.append([w, h])

        points = np.array(points, dtype=np.float32)  # [N_boxes,2]
        whs = np.array(whs, dtype=np.float32)  # [N_boxes,2]
        img = Image.open(os.path.join(self.img_path, img_file))
        img = img.convert("RGB")

        img_w, img_h = img.size
        orig_size = np.array([img_w, img_h])

        resize_w = 32 * int(img_w / 32)
        resize_h = 32 * int(img_h / 32)
        image = img.resize((resize_w, resize_h), Image.BILINEAR)
        image = self.transform(image)

        img_res = np.array([img_w, img_h], dtype=np.float32)

        scaled_whs = whs / img_res[None, :]

        anchor_points = points / img_res[None, :]
        all_points = all_points / img_res[None, :]
        labels = np.zeros(points.shape[0], dtype=np.int64)

        img_res = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)

        # img_name = int(img_file.split(".")[0])
        img_name = img_file

        sample = {
            "img_name": img_name,
            "image": image,
            "anchor_points": anchor_points,
            "whs": scaled_whs,
            "labels": labels,
            "orig_size": orig_size,
            "points": all_points,
        }
        return sample


def build_fscd_lvis(args, image_set):
    return FSCD_LVIS_Exemplars(args, image_set)


def build_fscd_lvis_points(args, image_set):
    return FSCD_LVIS_Points(args, image_set)


def build_fscd_lvis_test(args, image_set):
    return FSCD_LVIS_Test(args, image_set)
