import argparse
import os
import os.path as osp
from multiprocessing import Pool

import cv2


parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
parser.add_argument(
    "-i1", "--detection_folder", type=str, default="./debug_image/det_keep_score_text/",
)
parser.add_argument(
    "-i2", "--heatmap_folder", type=str, default="./debug_image/counting_keep_score_text/",
)
parser.add_argument(
    "-o", "--output_folder", type=str, default="./debug_image/combined_text/",
)
args = parser.parse_args()


def convert(args, img_name):
    output_path = osp.join(args.output_folder, img_name)
    det_image = cv2.imread(osp.join(args.detection_folder, img_name))
    heatmap_image = cv2.imread(osp.join(args.heatmap_folder, img_name))
    debug_image = cv2.hconcat((det_image, heatmap_image),)
    cv2.imwrite(output_path, debug_image)
    return


def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    list_images = next(os.walk(args.detection_folder))[2]
    pool = Pool(processes=10)
    for img_name in list_images:
        convert(args, img_name)


if __name__ == "__main__":
    main(args)
