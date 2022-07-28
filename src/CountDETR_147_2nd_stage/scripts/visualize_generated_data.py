import argparse
import os
from os.path import join

import cv2
from pycocotools.coco import COCO


def first_visualize_data(json_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    coco = COCO(json_path)
    img_ids = coco.getImgIds()
    for img_id in img_ids:
        anno_ids = coco.getAnnIds([img_id])
        img_info = coco.loadImgs([img_id])
        annos = coco.loadAnns(anno_ids)
        img_name = img_info[0]["file_name"]
        img_path = join("./FSC147/images_384_VarV2", img_name)
        img = cv2.imread(img_path)
        for anno in annos:
            box = anno["bbox"]
            x_cen, y_cen, w, h = box
            xmin, ymin, = x_cen - w / 2, y_cen - h / 2
            pred_box = [int(xmin), int(ymin), int(w), int(h)]
            pred_x, pred_y, pred_w, pred_h = pred_box
            img = cv2.rectangle(img, (pred_x, pred_y), (pred_x + pred_w, pred_y + pred_h), (0, 255, 0), 1)
        output_path = join(output_folder, img_name)
        cv2.imwrite(output_path, img)
    return


def second_visualize_data(json_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    coco = COCO(json_path)
    img_ids = coco.getImgIds()
    for img_id in img_ids:
        anno_ids = coco.getAnnIds([img_id])
        img_info = coco.loadImgs([img_id])
        annos = coco.loadAnns(anno_ids)
        img_name = img_info[0]["file_name"]
        img_path = join("./FSC147/images_384_VarV2", img_name)
        img = cv2.imread(img_path)
        for anno in annos:
            box = anno["bbox"]
            x1, y1, x2, y2 = box
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        output_path = join(output_folder, img_name)
        cv2.imwrite(output_path, img)
    return


def get_args_parser():
    parser = argparse.ArgumentParser("Visulize data", add_help=False)
    parser.add_argument("--json_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args_parser()
    first_visualize_data(args.json_path, args.output_dir)
    # second_visualize_data(args.json_path, args.output_dir)
