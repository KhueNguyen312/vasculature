import sys
sys.path.append("../vasculature")
# sys.path.append("ultralytics")
# sys.path.append("tools")
import os
from glob import glob
import torch.nn as nn
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tool_utils.ops import non_max_suppression
try:
    from segmentation.models import build_model
except ImportError as e:
    print(e)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_unet_model(checkpoint):
    #  '/kaggle/input/weights/best__unet2d__resnet34_f0.pth'
    backbone = os.path.basename(checkpoint).split("__")[-1]
    backbone = os.path.splitext(backbone)[0]
    # resnet34_f0 -> resnet34
    backbone = backbone[:-3]
    model = build_model('unet2d', backbone, pretrained=False)
    ckpt = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def rescale_masks(masks, h, w):
    """
    Rescale mask from any size to h, w
    """
    if masks is None:
        return np.zeros((1, h, w))
    masks = F.interpolate(masks.unsqueeze(0), size=[512, 512], mode='bilinear')[0].detach().cpu().numpy()
    return masks


def infer_yolo8(model, img_dir, save_path="fold0.json"):
    import json
    from tqdm import tqdm
    preds = []
    with torch.no_grad():
        for img in tqdm(glob(os.path.join(img_dir, "*"))):
            img_id = os.path.splitext(os.path.basename(img))[0]
            item = {
                "image_id": img_id,
                "annotations": []
            }
            pred = model.predict(img, conf=0.1, iou=0.5, verbose=False, save=False)[0]
            if len(pred.boxes):
                bboxes = pred.boxes.data.cpu().numpy()[:, :4].astype(int).tolist()
                labels = pred.boxes.cls.cpu().numpy().astype(int).tolist()
                confs = pred.boxes.conf.cpu().numpy().tolist()
            else:
                bboxes = []
                labels = []
                confs = []
            for b, l, c in zip(bboxes, labels, confs):
                item["annotations"].append({
                    "box": b,
                    "label": l,
                    "conf": c
                })
            preds.append(item)
    with open(save_path, "w") as f:
        json.dump(preds, f, indent=2)
    return preds

def infer_unet(model_seg, image, bboxes, img_size=224):
    """Inference cropped box with unet"""
    if len(bboxes) == 0:
        return np.zeros([1, *image.shape[:2]])

    images = []
    for x1, y1, x2, y2 in bboxes:
        crop = image[y1:y2, x1:x2]
        images.append(cv2.resize(crop, (img_size, img_size)))
    batch_img = np.stack(images).transpose(0, 3, 1 ,2)
    batch_img = torch.from_numpy(batch_img).to(device)
    masks = model_seg(batch_img / 255.0)

    masks = F.sigmoid(masks).detach().cpu().numpy()
    full_masks =[]
    for i in range(len(masks)):
        x1, y1, x2, y2 = bboxes[i]
        w, h = x2 - x1, y2 - y1        
        #cv2.imwrite(f"runs/masks_{i}.jpg", masks[i][0])
        mask_resize = cv2.resize(masks[i][0], (w, h))
        #cv2.imwrite(f"runs/masks_r{i}.jpg", mask_resize)
        full_mask = np.zeros(image.shape[:2])
        full_mask[y1:y2, x1:x2] = mask_resize
        full_masks.append(full_mask)
    return np.stack(full_masks, 0)


class TileInference:
    """Apply tile inference on single image, det only"""

    def __init__(self, tile=2):
        self.grids = make_grid(512, 512, tile)

    def _infer(self, model, img, imgsz=512, conf=0.24, iou=0.7):
        pred = model(img, imgsz=imgsz, conf=conf, iou=iou)[0]
        bboxes = pred.boxes.data.cpu().numpy()[:, :4].astype(int)
        labels = pred.boxes.cls.cpu().numpy().astype(int)
        confs = pred.boxes.conf.cpu().numpy()
        return bboxes, labels, confs

    def __call__(self, model, img,  model_full=None, imgsz=512, conf=0.24, iou=0.7):
        # Model_full, run on full image
        all_boxes = []
        all_labels = []
        all_confs = []
        for i, (x1, x2, y1, y2) in enumerate(self.grids):
            img_cut = img[y1:y2, x1:x2]
            boxes, labels, confs = self._infer(model, img_cut, imgsz=imgsz, conf=conf, iou=iou)
            boxes[:, [0, 2]] += x1  # shift x
            boxes[:, [1, 3]] += y1  # shift y
            all_boxes.append(boxes)
            all_labels.append(labels)
            all_confs.append(confs)
        
        if model_full is not None:
            boxes, labels, confs = self._infer(model, img, imgsz=imgsz, conf=conf, iou=iou)
            all_boxes.append(boxes)
            all_labels.append(labels)
            all_confs.append(confs)

        all_boxes = np.concatenate(all_boxes)
        all_confs = np.concatenate(all_confs)
        all_labels = np.concatenate(all_labels)
        
        keep_idx = non_max_suppression(
            all_boxes,
            all_confs,
            threshold=0.5
        )
        return all_boxes[keep_idx], all_labels[keep_idx], all_confs[keep_idx]