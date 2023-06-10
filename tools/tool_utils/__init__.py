import json
import os
import base64
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import _mask as coco_mask
import typing as t
import zlib
import cv2
import json
import torch

try:
    from shapely.geometry.polygon import Polygon
    import rasterio.features
except:
    pass

from itertools import product


def make_grid(H, W, cuts=4):
    # Code stolen from https://github.com/CarnoZhao/mmdetection/blob/sartorius_solution/sartorius/data.ipynb
    grids = []
    wstarts = W * np.arange(cuts).astype(int) // (cuts + 1)
    wends = W * np.arange(2, cuts + 2).astype(int) // (cuts + 1)
    hstarts = H * np.arange(cuts).astype(int) // (cuts + 1)
    hends = H * np.arange(2, cuts + 2).astype(int) // (cuts + 1)
    for i, j in product(range(cuts), range(cuts)):
        y1,y2,x1,x2, = hstarts[i], hends[i],wstarts[j],wends[j]
        grids.append([x1,x2,y1,y2])
    return grids


def load_annotations(path, indexing=False):
    anns = [] if not indexing else {}
    with open(path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        # print(result.keys(), result['id'], result['annotations'][1]['coordinates'])
        if not indexing:
            anns.append(result)
        else:
            anns[result['id']] = result
    return anns


def get_mask(image_id, annotations, to_masks=True):
    anns = annotations.get(image_id)
    if not anns:
        return

    if to_masks:
        # Convert to binary masks
        masks = []
        for ann in anns['annotations']:
            if ann['type'] != 'blood_vessel':
                continue
            poly = Polygon(ann['coordinates'][0])
            mask = rasterio.features.rasterize(
                [poly],
                out_shape=(512, 512)  # All data are 512x512
            )
            masks.append(mask)
        return masks

    return ann

def encode_binary_mask(mask: np.ndarray) -> t.Text:
    """Converts a binary mask into OID challenge encoding ascii text."""

    # check input mask --
    if mask.dtype != bool:
        raise ValueError(
            "encode_binary_mask expects a binary mask, received dtype == %s" %
            mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            "encode_binary_mask expects a 2d mask, received shape == %s" %
            mask.shape)

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str


def find_masks_watershed(mask, separate=True):
    # Code stolen from https://github.com/bnsreenu/python_for_microscopists/blob/master/205_predict_unet_with_watershed_single_image.py
    mask_rgb = np.stack([mask, mask, mask], -1)
    ret1, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    sure_bg = cv2.dilate(opening, kernel, iterations=10)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret2, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret3, markers = cv2.connectedComponents(sure_fg)
    markers[unknown==255] = 0
    labels = cv2.watershed(mask_rgb, markers)
    if DEBUG:
        print(np.unique(labels))
    if separate:
        used_labels = np.unique(labels)[2:]  # skip bg
        nb_masks = len(used_labels)
        rows, cols = mask.shape[:2]
        masks = np.zeros((nb_masks, rows, cols), dtype=bool)
        for i, l in enumerate(used_labels):
            masks[i] = labels == l
        return masks
    else:
        return labels

def draw_bboxes(img, bboxes, color=(255, 0, 0), thickness=2, labels=None, label=""):
    if labels is None:
        labels = [""] * len(bboxes)

    for (x1, y1, x2, y2), l in zip(bboxes, labels):
        cv2.rectangle(
            img,
            (int(x1), int(y1)), (int(x2), int(y2)),
            color,
            thickness
        )
        if isinstance(l, float):
            l = round(l, 2)
        text = f"{label}-{l}"
        cv2.putText(img, text, (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return img

def save_image_result(img, masks, gt_mask=None, path=None):
    mask = np.logical_or.reduce(masks) * 255
    if gt_mask is not None and isinstance(gt_mask, list):
        gt_mask = np.logical_or.reduce(gt_mask) * 255

    plt.figure(figsize=(15, 7))
    n_rows = 3 if gt_mask is not None else 2
    plt.subplot(1, n_rows, 1)
    plt.imshow(img)
    plt.title('Image')
    plt.axis('off')
    
    plt.subplot(1, n_rows, 2)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.4, cmap='inferno')
    plt.title('Prediction')
    plt.axis('off')
    
    if gt_mask is not None:
        plt.subplot(1, n_rows, 3)
        plt.imshow(img)
        plt.imshow(gt_mask, alpha=0.4, cmap='inferno')
        plt.title('Ground truth')
        plt.axis('off')
        plt.tight_layout()

    if path:
        plt.savefig(path)
        plt.clf()
    else:
        plt.show()


def encode_masks(masks, scores):
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
        scores = scores.cpu().numpy()

    pred_str = []
    for mask, conf in zip(masks, scores):
        mask = mask.astype(bool)
        encoded = encode_binary_mask(mask)
        pred_str.append(f"0 {conf} {encoded.decode('utf-8')}")
    return " ".join(pred_str)

def auto_img_dir():
    for p in [
        "/kaggle/input/hubmap-hacking-the-human-vasculature/train",
        "/data-storage/hubmap_data/train",
        "/content/data/train",
        "data/vasculature/train"
    ]:
        if os.path.exists(p):
            return p
    raise RuntimeError(f"Cannot find auto img_dir, please specify")

def auto_ann_path():
    for p in [
        "/kaggle/input/hubmap-hacking-the-human-vasculature/polygons.jsonl",
        "/data-storage/hubmap_data/polygons.jsonl",
        "/content/data/polygons.jsonl",
        "data/vasculature/polygons.jsonl",
    ]:
        if os.path.exists(p):
            return p
    raise RuntimeError(f"Cannot find auto ann_path, please specify")

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    anns = load_annotations('data/vasculature/polygons.jsonl', True)
    m = get_mask('89676f94fc96', anns)
    print(len(m))
    plt.imshow(np.logical_or.reduce(m) * 255)
    plt.show()