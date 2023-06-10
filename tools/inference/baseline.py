## Toolkit for running Inference and local validation
## 

import sys
sys.path.append("segmentation")
sys.path.append("ultralytics")
sys.path.append("tools")
import torch
import cv2
import os
import torch.nn.functional as F
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from models import build_model
from ultralytics import YOLO
from tqdm import tqdm
from tool_utils import load_annotations, get_mask, save_image_result
from ensemble import Ensemble
from skimage import measure
from pycocotools import mask as mask_utils
from shapely.geometry.polygon import Polygon
import rasterio.features
import json
from tool_utils.model import TileInference


def parse_args():
    parser = argparse.ArgumentParser("Inference 2 stage")
    parser.add_argument("--det_weight", type=str,
        help="Weight path of detection model")
    parser.add_argument("--seg_weight", type=str,
        help="Weight path of segmentation model")
    parser.add_argument("--img_dir", type=str,
        help="Path to image dir",
        default="/data-storage/hubmap_data/fold0/val")
    
    # Required for running validation
    parser.add_argument("--data_csv", type=str,
        help="Path to data.csv (splitted k fold)",
        default="")
    parser.add_argument("--ann_path", type=str,
        help="Path to polygons.jsonl", default="")
    parser.add_argument("--out", type=str,
        help="output directory", default="inference_output")
    parser.add_argument("--no_img", action="store_true",
        help="No save image")

    return parser.parse_args()


def make_seg(masks):
    """Convert binmask to polygon"""
    segmentations = []
    for mask in masks:
        segmentation = []
        # fortran_ground_truth_binary_mask = np.asfortranarray(mask)
        # encoded_ground_truth = mask_utils.encode(fortran_ground_truth_binary_mask)
        # ground_truth_area = mask_utils.area(encoded_ground_truth)
        # ground_truth_bounding_box = mask_utils.toBbox(encoded_ground_truth)
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation += contour.astype(int).ravel().tolist()

        # Revert
        # poly = Polygon( np.array(segmentation).astype(int).reshape(-1, 2))
        # maskr = rasterio.features.rasterize(
        #     [poly],
        #     out_shape=(512, 512)
        # )
        # print((maskr == mask).mean(), "??")
        segmentations.append(segmentation)
    return segmentations

if __name__ == "__main__":
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"RUNNING ON {device}")

    img_paths = glob(os.path.join(args.img_dir, "*"))

    # ense_model = Ensemble([
    #     YOLO(args.det_weight)
    # ])

    model = YOLO(args.det_weight)
    if args.seg_weight:
        model_seg = get_unet_model(args.seg_weight)
    else:
        model_seg = None

    os.makedirs(args.out, exist_ok=True)
    if not args.no_img:
        save_img_dir = os.path.join(args.out, "images")
        os.makedirs(save_img_dir, exist_ok=True)

    if args.ann_path:
        annotations = load_annotations(args.ann_path, indexing=True)
    else:
        annotations = {}

    # Do validation if annotation and data_csv is given
    do_valid = args.ann_path and args.data_csv

    thr = 0.5
    predictions = []
    for img_path in tqdm(img_paths):
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # inp_img = cv2.resize(img, (768, 768))
        # inp_img = torch.from_numpy(inp_img).permute(2, 0, 1).to(device).unsqueeze(0)

        pred_instances = model.predict(img, conf=0.5)[0]
        masks_yolo = getattr(pred_instances.masks, "data", None)
        masks_yolo = rescale_masks(masks_yolo, h, w)

        if model_seg is not None:
            masks_unet = infer_unet(
                model_seg,
                img,
                pred_instances.boxes.data.cpu().numpy()[:, :4].astype(int))

            # masks = (masks_yolo + masks_unet) / 2
            masks = masks_unet
        else:
            masks = masks_yolo

        masks = (masks > thr).astype(np.uint8)
        if not args.no_img:
            save_image_result(
                img, masks,
                # np.logical_or.reduce(masks_yolo) * 255,
                # get_mask(img_id, annotations, to_masks=True),
                path=os.path.join(save_img_dir, f"{img_id}.jpg")
            )


        for mask in make_seg(masks):
            predictions.append({
                "image_id_": img_id,
                "segmentation": mask
            })
    
    with open(os.path.join(args.out, "pred.json"), "w") as f:
        json.dump(predictions, f)
    # sub_df = pd.DataFrame(predictions, columns=['id', 'height', 'width', 'prediction_string'])
    # sub_df.head(1)
