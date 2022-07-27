import json
import os
import os.path as osp
import random as rd
import xml.etree.cElementTree as ET

import cv2
import numpy as np
from tqdm import tqdm


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

    annotation_files = osp.join(data_folder, "annotation_FSC147_384.json")
    with open(annotation_files, "r") as fp:
        annotations = json.load(fp)
    image_files = image_files[:20]
    for image_file in tqdm(image_files):
        try:
            image_path = osp.join(input_folder, image_file)
            image = cv2.imread(image_path)
            basename = image_file[:-4]
            xml_path = osp.join(data_folder, "xml_format", split, basename + ".xml")
            if not osp.isfile(xml_path):
                continue
            points, boxes = get_point_box(xml_path)
            to_use_box = boxes[0]
            x1, y1, x2, y2 = to_use_box
            box_w, box_h = x2 - x1, y2 - y1
            im_h, im_w, im_c = image.shape
            img = np.zeros([im_h, im_w, im_c], dtype=np.uint8)
            img.fill(255)  # or img[:] = 255
            img = cv2.rectangle(img, (0, 0), (im_w, im_h), (255, 0, 0), 4)
            for point in points:
                x, y = point
                low_w, high_x = int(x - box_w / 2), int(x + box_w / 2)
                low_h, high_h = int(y - box_h / 2), int(y + box_h / 2)
                xs = rd.sample(range(low_w, high_x), 2)
                ys = rd.sample(range(low_h, high_h), 2)
                for (x, y) in zip(xs, ys):
                    img = cv2.circle(img, (int(x), int(y)), 2, (0, 0, 0), 2)
            # new_img = cv2.hconcat([image, img])
            new_img = img
            new_path = osp.join(to_write_folder, image_file)
            cv2.imwrite(new_path, new_img)
        except Exception as e:
            print(image_file)
            print(points, boxes)
            continue
    return


if __name__ == "__main__":
    input_folder = "./FSC147"
    output_folder = "./debug_image/point_sampling/"
    os.makedirs(output_folder, exist_ok=True)
    for split in [
        "test",
    ]:
        print(split)
        vis_data(input_folder, output_folder, split=split)
