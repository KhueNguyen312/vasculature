
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
import glob
# from pycocotools import mask
from shapely.geometry.polygon import Polygon
# from mmengine.fileio import dump, list_from_file
# from mmengine.utils import mkdir_or_exist, scandir, track_iter_progress
from tqdm import tqdm
from PIL import Image
from tool_utils import load_annotations


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
        default='data/vasculature/raw/polygons.jsonl'
    )
    parser.add_argument(
        '--out',
        type=str,
        help='output directory')
    parser.add_argument(
        '--fold',
        type=int,
        default=0
    )
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
            
def cvt_to_coco_json(img_infos, label_map, annotations, image_ids, ignore_classes):
    image_ids = set(image_ids)
    ignore_classes = set(ignore_classes)

    image_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    for category_id, name in label_map.items():
        if str(name) in ignore_classes:
            continue
        # treat unsure as blood vessel
        if name == 'unsure':
            name = 'blood_vessel'
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    img_map = {}
    # COCO use int as image_id
    image_strid_2_intid = {}
    for i, img_dict in enumerate(img_infos):
        file_name = os.path.basename(img_dict['filename'])
        image_id = os.path.splitext(file_name)[0]
        if image_id not in image_ids:
            continue
        # print(image_id)
        if file_name in image_set:
            print(f"Duplicate image {file_name}", img_dict)
        image_item = dict()
        image_item['id'] = i
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(img_dict['height'])
        image_item['width'] = int(img_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)
        img_map[image_id] = image_item
        image_strid_2_intid[image_id] = i

    i = 0
    for anns in tqdm(annotations):
        image_id = anns['id']
        if image_id not in image_ids:
            continue
        for ann in anns['annotations']:
            if image_id not in img_map:
                # print(f"WARNING image {image_id} not found")
                continue
            if ann['type'] in ignore_classes:
                continue

            # treat unsure as blood vessel
            if ann['type'] == 'unsure':
                ann['type'] =  'blood_vessel'

            img = img_map[image_id]
            ann_item = dict()
            ann_item['id'] = i
            ann_item['image_id'] = image_strid_2_intid[image_id]
            ann_item['image_id_'] = image_id
            ann_item['iscrowd'] = 0
            ann_item['segmentation'] = make_poly(ann['coordinates'])
            img = img_map[image_id]
            poly = Polygon(ann['coordinates'][0])
            x1, y1, x2, y2 = poly.bounds
            ann_item['bbox'] = [int(x) for x in [x1, y1, x2 - x1, y2 - y1]]
            ann_item['area'] = poly.area
            ann_item['category_id'] = inverted_label_map[ann['type']]
            coco['annotations'].append(ann_item)
            i += 1
    return coco


def copy_images(img_dir, image_ids, save_dir):
    image_ids = set(image_ids)
    os.makedirs(save_dir, exist_ok=True)

    for img_path in glob.glob(os.path.join(img_dir, "*")):
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        if  image_id in image_ids:
            shutil.copy(img_path, os.path.join(save_dir, os.path.basename(img_path)))


def main():
    args = parse_args()
    out_dir = args.out#os.path.join(args.out, f"fold_{args.fold}")
    # if os.path.isdir(out_dir):
    #     import shutil
    #     shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv("data.csv")
    train_df = df[df["fold"] != args.fold]
    val_df = df[df["fold"] == args.fold]
    has_ann_ids = df['id'].tolist()

    print(f"Train/val {len(train_df)}/{len(val_df)}, {len(has_ann_ids)}")

    # 1 load image list info
    img_infos = collect_image_infos(args.img_path, None, has_ann_ids=has_ann_ids)

    # Load annotations
    annotations = load_annotations(args.ann_path)
    # 2 convert to coco format data
    train_ids = train_df['id'].tolist()
    val_ids = val_df['id'].tolist()
    train_coco = cvt_to_coco_json(img_infos, label_map, annotations, image_ids=train_ids, ignore_classes=args.ignore_classes)
    val_coco = cvt_to_coco_json(img_infos, label_map, annotations, image_ids=val_ids, ignore_classes=args.ignore_classes)
    if not args.no_image:
        print("Copying images ...")
        copy_images(args.img_path, train_ids, os.path.join(out_dir, 'train'))
        copy_images(args.img_path, val_ids, os.path.join(out_dir, 'val'))
    # 3 dump
    dump(train_coco, os.path.join(out_dir, "annotations_train.json"))
    dump(val_coco, os.path.join(out_dir, "annotations_valid.json"))
    print(f'save json files: {out_dir}')


if __name__ == '__main__':
    main()
