import json
import os

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms as transforms


class FSCD_LVISDataset(Dataset):
    def __init__(
        self, args, split="train",
    ):
        data_path = args.data_path
        pseudo_label_file = "pseudo_lvis_" + split + "_cxcywh.json"
        self.coco = COCO(os.path.join(data_path, "annotations_old", pseudo_label_file))
        self.image_ids = self.coco.getImgIds()

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        self.img_path = os.path.join(data_path, "images", "all_images")
        self.count_anno_file = os.path.join(data_path, "annotations_old", "count_" + split + ".json")
        self.count_anno = self.load_json(self.count_anno_file)

    def load_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.image_ids)

    def get_gt(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_file = img_info["file_name"]
        img = Image.open(os.path.join(self.img_path, img_file))
        img = img.convert("RGB")

        wh = img.size
        ann_ids = self.coco.getAnnIds([img_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        bboxes = np.array([instance["bbox"] for instance in anns], dtype=np.float32)

        ex_bboxes = self.count_anno["annotations"][idx]["boxes"]

        ex_rects = []
        for bbox in ex_bboxes[:3]:
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            ex_rects.append([x1, y1, x2, y2])

        ex_rects = np.array(ex_rects, dtype=np.float32)
        ex_rects[:, 0] = np.clip(ex_rects[:, 0], 0, img.size[0] - 1)
        ex_rects[:, 1] = np.clip(ex_rects[:, 1], 0, img.size[1] - 1)
        ex_rects[:, 2] = np.clip(ex_rects[:, 2], 0, img.size[0] - 1)
        ex_rects[:, 3] = np.clip(ex_rects[:, 3], 0, img.size[1] - 1)
        return img, bboxes, ex_rects, wh

    def __getitem__(self, index):
        img, bboxes, ex_rects, wh = self.get_gt(index)
        img_w, img_h = img.size

        orig_size = np.array([img_h, img_w])

        resize_w = 32 * int(img_w / 32)
        resize_h = 32 * int(img_h / 32)
        img = img.resize((resize_w, resize_h))
        img = self.transform(img)

        img_res = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
        bboxes = bboxes.astype(np.float32) / img_res[None, :]

        ex_rects = ex_rects.astype(np.float32) / img_res[None, :]
        labels = torch.zeros([bboxes.shape[0]], dtype=torch.int64)

        ret = {
            "image": img,
            "boxes": bboxes,
            "ex_rects": ex_rects,
            "origin_wh": wh,
            "labels": labels,
            "orig_size": orig_size,
        }
        return ret


class FSCD_LVIS_Dataset_Test(Dataset):
    def __init__(
        self, args, split="test",
    ):
        super().__init__()
        print("This data is fscd 147 test set, split: {}".format(split))

        data_path = args.data_path
        pseudo_label_file = "single_instances_" + split + ".json"
        self.coco = COCO(os.path.join(data_path, "annotations_old", pseudo_label_file))
        self.image_ids = self.coco.getImgIds()

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        self.img_path = os.path.join(data_path, "images", "all_images")
        self.count_anno_file = os.path.join(data_path, "annotations_old", "count_" + split + ".json")
        self.count_anno = self.load_json(self.count_anno_file)

    def load_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.image_ids)

    def get_gt(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_file = img_info["file_name"]
        img = Image.open(os.path.join(self.img_path, img_file))
        img = img.convert("RGB")

        wh = img.size
        ann_ids = self.coco.getAnnIds([img_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        bboxes = np.array([instance["bbox"] for instance in anns], dtype=np.float32)

        ex_bboxes = self.count_anno["annotations"][idx]["boxes"]

        ex_rects = []
        for bbox in ex_bboxes[:3]:
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            ex_rects.append([x1, y1, x2, y2])

        ex_rects = np.array(ex_rects, dtype=np.float32)

        return img, bboxes, ex_rects, wh

    def __getitem__(self, index):
        img, bboxes, ex_rects, wh = self.get_gt(index)
        img_w, img_h = img.size

        orig_size = np.array([img_h, img_w])

        resize_w = 32 * int(img_w / 32)
        resize_h = 32 * int(img_h / 32)
        img = img.resize((resize_w, resize_h))
        img = self.transform(img)

        img_res = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
        bboxes = bboxes.astype(np.float32) / img_res[None, :]

        ex_rects = ex_rects.astype(np.float32) / img_res[None, :]
        labels = torch.zeros([bboxes.shape[0]], dtype=torch.int64)

        ret = {
            "image": img,
            "boxes": bboxes,
            "ex_rects": ex_rects,
            "origin_wh": wh,
            "labels": labels,
            "orig_size": orig_size,
        }
        return ret
