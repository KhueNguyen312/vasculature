import sys
sys.path.append("./tools")
import os
import matplotlib.pyplot as plt
import cv2
from tool_utils import load_annotations, draw_bboxes
import json
from tool_utils import auto_ann_path, auto_img_dir
from shapely.geometry import Polygon
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion as wbf


parser = argparse.ArgumentParser()
parser.add_argument('--oof_file', nargs="+", help="path to oof json file", type=str)
parser.add_argument('--out', type=str, default="ensemble.json")
args = parser.parse_args()

img_dir = auto_img_dir()
save_dir = "/u01/phatth1/visualization"
annotations = load_annotations(auto_ann_path(), indexing=True)
pred_path = "/u01/phatth1/runs/exp_yolov5l_det_768_f0/oof_fold0.json"
out = "/u01/phatth1"

def get_gt_bboxes(img_id):
    anns = annotations.get(img_id)
    bboxes = []

    for ann in anns["annotations"]:
        bbox = Polygon(ann["coordinates"][0]).bounds
        bbox = [int(x) for x in bbox]
        bboxes.append(bbox)
    return bboxes

with open(args.oof_file, "r") as f:
    preds = json.load(f)


## Merge box wbf


## PLOT Prediction vs Annotation
merged = []
for pred in tqdm(preds):
    img_id = pred["image_id"]
    item = {"image_id": img_id}
    img = cv2.imread(os.path.join(img_dir, f"{img_id}.tif"))
    # pred["annotations"] = [x for x in pred["annotations"] if x["conf"] >= 0.5]
    pred_bboxes = [a["box"] for a in pred["annotations"]]
    pred_confs= [a["conf"] for a in pred["annotations"]]
    gt_bboxes = get_gt_bboxes(img_id)

    item["bboxes"], item["confs"], item["labels"] = wbf(
        [pred_bboxes, gt_bboxes],
        [pred_confs, [0.95] * len(gt_bboxes)],
        [[0] * len(pred_bboxes), [0] * len(gt_bboxes)]
    )
    # SPLOT
    # draw_bboxes(img, pred_bboxes, (255, 0, 0), labels=pred_confs, label="pred")
    # draw_bboxes(img, gt_bboxes , (0, 244, 10), label="gt")
    # cv2.imwrite(os.path.join(save_dir, f"{img_id}.jpg"), img)