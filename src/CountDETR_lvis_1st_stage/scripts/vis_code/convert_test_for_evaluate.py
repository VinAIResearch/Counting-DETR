import json
import os

ori_path = "./FSCD_LVIS/instances_test.json"
ori_test = json.load(open(ori_path, "r"))
new_test = ori_test.copy()
new_test["annotations"] = list()
old_test_annos = ori_test["annotations"]

for anno in old_test_annos:
    new_anno = anno.copy()
    new_anno["category_id"] = 1
    new_test["annotations"].append(new_anno)

new_path = "./FSCD_LVIS/single_cls_instances_test.json"
with open(new_path, "w") as handle:
    json.dump(new_test, handle)