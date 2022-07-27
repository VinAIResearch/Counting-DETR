import json
import os

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms as transforms


class FSC147Dataset(Dataset):
    def __init__(
        self, args, split="train",
    ):
        super().__init__()
        data_path = args.data_path
        pseudo_label_file = "pseudo_bbox_" + split + ".json"
        self.coco = COCO(os.path.join(data_path, "annotations", pseudo_label_file))
        self.images = self.coco.getImgIds()

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        self.img_path = os.path.join(data_path, "images_384_VarV2")
        self.anno_file = os.path.join(data_path, "annotation_FSC147_384.json")
        self.annotations = self.load_json(self.anno_file)

        self.density_path = os.path.join(data_path, "centerness_map")

    def load_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)

        return data

    def __len__(self):
        return len(self.images)

    def get_gt(self, idx):
        img_id = self.images[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_file = img_info["file_name"]
        img = Image.open(os.path.join(self.img_path, img_file))
        wh = img.size
        ann_ids = self.coco.getAnnIds([img_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        bboxes = np.array([instance["bbox"] for instance in anns], dtype=np.float32)

        anno = self.annotations[img_file]
        ex_bboxes = anno["box_examples_coordinates"]

        ex_rects = []
        for bbox in ex_bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            ex_rects.append([x1, y1, x2, y2])

        ex_rects = np.array(ex_rects, dtype=np.float32)
        # density_file = os.path.join(self.density_path,img_file.split('.')[0]+'.npy')
        # centerness_map = np.load(density_file)

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

        xyxx_boxes = np.zeros(bboxes.shape, dtype=np.float32)
        xyxx_boxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
        xyxx_boxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
        xyxx_boxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2
        xyxx_boxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2

        ret = {
            "image": img,
            "boxes": bboxes,
            "ex_rects": ex_rects,
            "origin_wh": wh,
            "labels": labels,
            "orig_size": orig_size,
            "xyxy_boxes": xyxx_boxes,
            # "centerness_map": centerness_map,
        }
        return ret


class FSC147_Dataset_Val(Dataset):
    def __init__(self, args, split="val", sample_point=False):
        print("This data is fscd 147 validation set, split: {}".format(split))
        data_path = args.data_path
        self.anno_file = os.path.join(data_path, "annotation_FSC147_384.json")
        self.data_split_file = os.path.join(data_path, "Train_Test_Val_FSC_147.json")
        self.im_dir = os.path.join(data_path, "images_384_VarV2")
        self.val_file = os.path.join(data_path, "instances_val.json")
        self.scale_factor = args.scale_factor

        self.annotations = self.load_json(self.anno_file)
        self.data_split = self.load_json(self.data_split_file)[split]
        self.label_val = COCO(self.val_file)
        self.img_name_to_ori_id = self.map_img_name_to_ori_id()
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        self.sample_point = sample_point
        self.density_path = os.path.join(data_path, "centerness_map")

        if self.sample_point:
            self.num_samples = 300

        print("This data contains: {} images".format(len(self.data_split)))
        self.density_path = os.path.join(data_path, "centerness_map")

    def load_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)

        return data

    def __len__(self):
        return len(self.data_split)

    def map_img_name_to_ori_id(self,):
        all_coco_imgs = self.label_val.imgs
        map_name_2_id = dict()
        for k, v in all_coco_imgs.items():
            img_id = v["id"]
            img_name = v["file_name"]
            map_name_2_id[img_name] = img_id
        return map_name_2_id

    def __getitem__(self, idx):
        img_name = self.data_split[idx]
        coco_im_id = self.img_name_to_ori_id[img_name]
        anno_ids = self.label_val.getAnnIds([coco_im_id])
        annos = self.label_val.loadAnns(anno_ids)

        box_centers = list()
        whs = list()
        xyxy_boxes = list()
        for anno in annos:
            bbox = anno["bbox"]
            x1, y1, w, h = bbox
            box_centers.append([x1 + w / 2, y1 + h / 2])
            whs.append([w, h])
            xyxy_boxes.append([x1, y1, x1 + w, y1 + h])

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
        orig_size = np.array([img_h, img_w])
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

        xyxy_boxes = np.array(xyxy_boxes, dtype=np.float32)
        img_res = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
        xyxy_boxes = (
            xyxy_boxes / img_res[None,]
        )
        ret = {
            "image_id": coco_im_id,
            "image": image,
            "points": scaled_points,
            "boxes": scaled_boxes,
            "orig_size": orig_size,
            "exemplar_boxes": scaled_exemplar_boxes,
            "labels": labels,
            "xyxy_boxes": xyxy_boxes,
        }
        return ret


class FSC147_Dataset_Test(Dataset):
    def __init__(self, args, split="test", sample_point=False):
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
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        self.sample_point = sample_point
        self.density_path = os.path.join(data_path, "centerness_map")

        if self.sample_point:
            self.num_samples = 300

        print("This data contains: {} images".format(len(self.data_split)))
        self.density_path = os.path.join(data_path, "centerness_map")

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

    def get_sample_points(
        self, density_map_file,
    ):
        density_map = np.load(density_map_file)

        h, w = density_map.shape

        array_density = np.reshape(density_map, (w * h))
        sum_value = np.sum(array_density)
        distribution = array_density / sum_value
        x_pos = np.arange(0, w)
        y_pos = np.arange(0, h)
        xv, yv = np.meshgrid(x_pos, y_pos)
        xv, yv = np.reshape(xv, (w * h)), np.reshape(yv, (w * h))
        indexes = np.arange(0, w * h)

        selected_indexes = np.random.choice(indexes, size=self.num_samples, p=distribution)

        selected_x_pos = xv[selected_indexes]
        selected_y_pos = yv[selected_indexes]
        scaled_x = selected_x_pos / w
        scaled_y = selected_y_pos / h

        selected_points = np.vstack((scaled_x, scaled_y))

        return selected_points

    def __getitem__(self, idx):
        img_name = self.data_split[idx]
        coco_im_id = self.img_name_to_ori_id[img_name]
        anno_ids = self.label_test.getAnnIds([coco_im_id])
        annos = self.label_test.loadAnns(anno_ids)

        box_centers = list()
        whs = list()
        xyxy_boxes = list()
        for anno in annos:
            bbox = anno["bbox"]
            x1, y1, w, h = bbox
            box_centers.append([x1 + w / 2, y1 + h / 2])
            whs.append([w, h])
            xyxy_boxes.append([x1, y1, x1 + w, y1 + h])

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
        orig_size = np.array([img_h, img_w])
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
        # density_file = os.path.join(self.density_path, img_name.split('.')[0]+'.npy')
        # centerness_map = np.load(density_file)

        # if not self.sample_point:
        xyxy_boxes = np.array(xyxy_boxes, dtype=np.float32)
        img_res = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
        xyxy_boxes = (
            xyxy_boxes / img_res[None,]
        )
        ret = {
            "image_id": coco_im_id,
            "image": image,
            "points": scaled_points,
            "boxes": scaled_boxes,
            "orig_size": orig_size,
            "exemplar_boxes": scaled_exemplar_boxes,
            "labels": labels,
            "xyxy_boxes": xyxy_boxes,
            # "centerness_map": centerness_map,
        }
        return ret
