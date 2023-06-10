"""
Code to covert competition data into cropped mask
for sematic segmentation model
"""


# Convert competition data into cooc format

# output:
#     <out_dir>/fold_<fold>:
#         annotations_train.json
#         annotations_val.json

import argparse
import os
import json
import pandas as pd
import numpy as np
import shutil
import cv2
import glob
# from pycocotools import mask
from shapely.geometry.polygon import Polygon
import rasterio.features
from tool_utils import load_annotations

# from mmengine.fileio import dump, list_from_file
# from mmengine.utils import mkdir_or_exist, scandir, track_iter_progress
from tqdm import tqdm
from PIL import Image

label_map = {
    1: 'blood_vessel',
    2: 'glomerulus',
    3: 'unsure'
}
inverted_label_map = {v: k for k, v in label_map.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert images to coco format without annotations')
    parser.add_argument('--img_path', help='The root path of images')
    parser.add_argument(
        '--ann_path',
        help='Path to annotation file, polygons.jsonl',
        default='data/vasculature/polygons.jsonl'
    )
    parser.add_argument(
        '--out',
        type=str,
        help='output directory')
    parser.add_argument(
        '--ignore_classes',
        nargs='+',
        default=[]
    )
    parser.add_argument(
        '--no_image',
        action='store_true',
        help='No save image'
    )
    parser.add_argument(
        '--crop',
        action='store_true',
        help='Crop by bounding boxes'
    )
    args = parser.parse_args()
    return args


def scandir(dir_path, suffix=None, recursive=False, case_sensitive=True):
    from pathlib import Path
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    if suffix is not None and not case_sensitive:
        suffix = suffix.lower() if isinstance(suffix, str) else tuple(
            item.lower() for item in suffix)

    root = dir_path

    def _scandir(dir_path, suffix, recursive, case_sensitive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = os.path.relpath(entry.path, root)
                _rel_path = rel_path if case_sensitive else rel_path.lower()
                if suffix is None or _rel_path.endswith(suffix):
                    yield rel_path
            elif recursive and os.path.isdir(entry.path):
                # scan recursively if entry.path is a directory
                yield from _scandir(entry.path, suffix, recursive,
                                    case_sensitive)

    return _scandir(dir_path, suffix, recursive, case_sensitive)

def dump(data, json_file):
    with open(json_file, "w") as f:
        json.dump(data, f)

def collect_image_infos(path, exclude_extensions=None, has_ann_ids=[]):
    img_infos = []
    has_ann_ids = set(has_ann_ids)
    used = set()
    images_generator = scandir(path, recursive=True)
    for image_path in list(images_generator):
        img_id = os.path.splitext(os.path.basename(image_path))[0]
        if img_id in has_ann_ids:
            image_path = os.path.join(path, image_path)
            img_pillow = Image.open(image_path)
            img_info = {
                'filename': image_path,
                'width': img_pillow.width,
                'height': img_pillow.height,
            }
            img_infos.append(img_info)
            used.add(img_id)
        elif img_id == '47cb7f4ec51f':
            print(image_path, img_id)
    print(f"Found {len(img_infos)} images")
    # print(has_ann_ids - used)
    return img_infos

def make_poly(coordinates):
    poly = []
    for c in coordinates:
        p = []
        for i in c:
            p += i
        poly.append(p)
    return poly

def path_2_id(img_path):
    file_name = os.path.basename(img_path)
    image_id = os.path.splitext(file_name)[0]
    return image_id

def make_masks(img_infos, label_map, annotations, ignore_classes, out_dir):
    ignore_classes = set(ignore_classes)

    image_set = set()
    img_map = {}
    # COCO use int as image_id
    for i, img_dict in enumerate(img_infos):
        file_name = os.path.basename(img_dict['filename'])
        image_id = os.path.splitext(file_name)[0]
        if file_name in image_set:
            print(f"Duplicate image {file_name}", img_dict)
        image_item = dict()
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(img_dict['height'])
        image_item['width'] = int(img_dict['width'])
        image_set.add(file_name)
        img_map[image_id] = image_item

    i = 0
    for anns in tqdm(annotations):
        image_id = anns['id']
        if image_id not in img_map:
            # print(f"WARNING image {image_id} not found")
            continue
        img_info = img_map[image_id]
        h, w = img_info['height'], img_info['width']
        full_mask = np.zeros((h, w), dtype=np.uint8)

        for ann in anns['annotations']:
            if ann['type'] in ignore_classes:
                # print(f"Skiping {ann['type']}")
                continue
            poly = Polygon(ann['coordinates'][0])
            mask = rasterio.features.rasterize(
                [poly],
                out_shape=(h, w)
            )
            # print(np.max(mask))
            mask *= 255

            full_mask = full_mask | mask

        cv2.imwrite(os.path.join(out_dir, f'{image_id}.png'), full_mask)
            # exit()

    return 1

def make_cropped_masks(img_infos, label_map, annotations, ignore_classes, out_dir, df):
    img_dir = os.path.join(out_dir, "images")
    mask_dir = os.path.join(out_dir, "masks")

    id2fold = dict(zip(df['id'], df['fold']))

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    ignore_classes = set(ignore_classes)

    # Mapping image_id -> polygon masks
    ann_map = {}
    img_map = {path_2_id(item['filename']): item for item in img_infos}
    # img_map = {'db8391d91d8d': img_map['db8391d91d8d']}

    i = 0
    box_df = []
    for anns in tqdm(annotations):
        image_id = anns['id']
        # All competition data is 512
        img_info = img_map.get(image_id)
        if not img_info:
            # print(f"{image_id} not found!!")
            continue
        h, w = img_info['height'], img_info['width']
        polys = []
        for ann in anns['annotations']:
            if ann['type'] in ignore_classes:
                continue
            poly = Polygon(ann['coordinates'][0])
            polys.append(poly)

        # Mem leak??
        ann_map[image_id] = polys

    # COCO use int as image_id
    for i, img_dict in enumerate(tqdm(img_infos)):
        file_name = os.path.basename(img_dict['filename'])
        image_id = os.path.splitext(file_name)[0]
        if image_id not in img_map:
            print(f"{image_id} not found!!")
            continue

        image = cv2.imread(img_dict['filename'])
        polys = ann_map.get(image_id)
        if polys is None:
            continue

        used = set()
        for i, poly in enumerate(polys):
            mask = rasterio.features.rasterize(
                [poly],
                out_shape=(img_dict['height'], img_dict['width'])
            )
            mask = (mask * 255).astype(np.uint8)
            x1, y1, x2, y2 = [int(x) for x in poly.bounds]
            k = (x1, y1, x2, y2)
            if k in used:
                print(f"WARN --> image_id: {k} duplicate!!!")
            # expand = 0.1 # 10%
            # Do crop
            crop_img = image[y1:y2, x1:x2]
            crop_mask = mask[y1:y2, x1:x2]
            
            try:
                cv2.imwrite(os.path.join(img_dir, f'{image_id}_{i}.jpg'), crop_img)
            except  Exception as e:
                print(crop_img.shape, x1,y1,x2,y2, expand)
                raise e
            cv2.imwrite(os.path.join(mask_dir, f'{image_id}_{i}.jpg'), crop_mask)
            box_df.append([image_id, f'{image_id}_{i}', x1, y1, x2 - x1, y2 - y1, id2fold[image_id]])

    pd.DataFrame(box_df, columns=['image_id', 'id', 'x', 'y', 'w', 'h', 'fold']).to_csv('bbox.csv', index=False)
    return 1


def copy_images(img_dir, image_ids, save_dir):
    image_ids = set(image_ids)
    os.makedirs(save_dir, exist_ok=True)

    for img_path in glob.glob(os.path.join(img_dir, "*")):
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        if  image_id in image_ids:
            shutil.copy(img_path, os.path.join(save_dir, os.path.basename(img_path)))


def main():
    args = parse_args()
    args.ignore_classes = ['glomerulus', 'unsure']
    out_dir = args.out#os.path.join(args.out, f"fold_{args.fold}")

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv("data.csv")
    has_ann_ids = df['id'].tolist()

    assert os.path.exists(args.img_path), f"{args.img_path} not found"

    # 1 load image list info
    img_infos = collect_image_infos(args.img_path, None, has_ann_ids=has_ann_ids)

    # Load annotations
    annotations = load_annotations(args.ann_path)
    # 2 convert to coco format data
    if args.crop:
        make_cropped_masks(img_infos, label_map, annotations, 
            ignore_classes=args.ignore_classes, out_dir=out_dir,
            df=df)
    else:
        make_masks(img_infos, label_map, annotations, 
            ignore_classes=args.ignore_classes, out_dir=out_dir)
    print(f'save masks files: {out_dir}')


if __name__ == '__main__':
    main()
