import argparse
import json
import os
import os.path as osp
import xml.etree.cElementTree as ET

import cv2
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
parser.add_argument("-dp", "--data_path", type=str, default="./FSC147/", help="Path to the FSC147 dataset")
parser.add_argument("--all_points", action="store_true", help="vis all points")
parser.add_argument("--all_boxes", action="store_true", help="vis all boxes")
parser.add_argument("--heat_map", action="store_true", help="vis heat map")
parser.add_argument("--example_boxes", action="store_true", help="vis example boxes")
parser.add_argument("--num_image", type=int, default=0, help="vis example boxes")

args = parser.parse_args()

data_path = args.data_path
anno_file = data_path + "annotation_FSC147_384.json"
data_split_file = data_path + "Train_Test_Val_FSC_147.json"
im_dir = data_path + "images_384_VarV2"
gt_dir = data_path + "gt_density_map_adaptive_384_VarV2"


def get_point_box(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    points = []
    for object_iter in root.findall("object"):
        bndbox = object_iter.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
        points.append([int((xmin + xmax) / 2), int((ymin + ymax) / 2)])
    return points, boxes


def vis_data(data_folder, output_folder, split="test"):
    input_folder = osp.join(data_folder, "images_384_VarV2")
    image_cls_files = osp.join(data_folder, "ImageClasses_FSC147.txt")
    to_write_folder = osp.join(output_folder, split)
    os.makedirs(to_write_folder, exist_ok=True)

    with open(image_cls_files, "r") as handle:
        lines = handle.readlines()
        lines = [l.rstrip() for l in lines]

    train_val_test_json_files = osp.join(data_folder, "Train_Test_Val_FSC_147.json")
    with open(train_val_test_json_files, "r") as handle:
        splits = json.load(handle)
    image_files = splits[split]
    if args.num_image != 0:
        image_files = image_files[: args.num_image]

    annotation_files = osp.join(data_folder, "annotation_FSC147_384.json")
    with open(annotation_files, "r") as fp:
        annotations = json.load(fp)

    for image_file in tqdm(image_files):
        image_path = osp.join(input_folder, image_file)
        image = cv2.imread(image_path)
        basename = image_file[:-4]
        xml_path = osp.join(data_folder, "xml_format", split, basename + ".xml")
        if not osp.isfile(xml_path):
            continue
        points, boxes = None, None
        if args.all_boxes:
            _, boxes = get_point_box(xml_path)

        if args.all_points:
            points = annotations[image_file]["points"]

        if args.example_boxes:
            bboxes = annotations[image_file]["box_examples_coordinates"]
            boxes = []
            for bbox in bboxes:
                x1 = bbox[0][0]
                y1 = bbox[0][1]
                x2 = bbox[2][0]
                y2 = bbox[2][1]
                boxes.append([x1, y1, x2, y2])

        if boxes is not None:
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        if points is not None:
            for point in points:
                x, y = point
                cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), 2)

        if args.heat_map:
            density_path = gt_dir + "/" + image_file.split(".jpg")[0] + ".npy"
            density = np.load(density_path).astype("float32")

            gt_heatmap = None
            gt_heatmap = cv2.normalize(
                density, gt_heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            gt_heatmap = cv2.applyColorMap(gt_heatmap, cv2.COLORMAP_JET)
            image = gt_heatmap

        new_path = osp.join(to_write_folder, image_file)
        cv2.imwrite(new_path, image)
    return


if __name__ == "__main__":
    input_folder = "./FSC147"
    output_folder = "./outputs/vis_exemplar_boxes/"
    os.makedirs(output_folder, exist_ok=True)
    for split in [
        "train",
        "val",
        "test",
    ]:
        print(split)
        vis_data(input_folder, output_folder, split=split)
