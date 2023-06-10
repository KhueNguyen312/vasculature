import shutil
import os
import pandas as pd
import numpy as np
import cv2
# import tifffile as tiff
import argparse
from pathlib import Path
import os
from glob import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tool_utils import load_annotations, auto_ann_path, auto_img_dir,make_grid
from shapely.geometry.polygon import Polygon
import rasterio.features

class_map = {
    'blood_vessel': 0,
    'unsure': 1,
    'glomerulus': 2,
}

def mkdir_yolo_data(train_path, val_path):
    train_image_path = Path(f'{train_path}/images')
    train_label_path = Path(f'{train_path}/labels')
    val_image_path = Path(f'{val_path}/images')
    val_label_path = Path(f'{val_path}/labels')
    
    train_image_path.mkdir(parents=True, exist_ok=True)
    train_label_path.mkdir(parents=True, exist_ok=True)
    val_image_path.mkdir(parents=True, exist_ok=True)
    val_label_path.mkdir(parents=True, exist_ok=True)
    
    return train_image_path, train_label_path, val_image_path, val_label_path

def create_vessel_annotations(polygons_path, use_classes=['blood_vessel']):
    # load polygons data
    polygons = pd.read_json(polygons_path, orient='records', lines=True)
    
    # extract blood_vessel annotation
    annotations_dict = defaultdict(list)
    
    for idx, row in polygons.iterrows():
        id_ = row['id']
        annotations = row['annotations']
        for annotation in annotations:
            if annotation['type'] in use_classes:
                annotations_dict[id_].append([annotation['coordinates'], annotation['type']])
    
    return annotations_dict

def create_label_file(id_, annotations, path, img_size, single_class, task):
    """
    Create label txt file for yolo v8
    
    parameters
    ----------
    id_: str
        label id
    annotations: list
        coordinates and type of annotation
    path: str
    single_class: bool if true use same class for blood_vessel and unsure
    task: seg or det
    
    path for saving label txt file
    """
    label_txt = ''
    
    for (coordinate, class_name) in annotations:
        # TODO: class id
        if single_class:
            label_txt += '0 '
        else:
            label_txt += f'{class_map[class_name]} '
        # Normalize
        if task == "seg":
            coor_array = np.array(coordinate[0]).astype(float)
            coor_array /= float(img_size)
            # transform to str
            coor_list = list(coor_array.reshape(-1).astype(str))
            coor_str = ' '.join(coor_list)
        else:
            # detection
            x1, y1, x2, y2 = Polygon(coordinate[0]).bounds
            w, h = x2 - x1, y2 - y1
            xc, yc = x1 + w / 2, y1 + h / 2
            coor_str = f'{xc/img_size} {yc/img_size} {w/img_size} {h/img_size}'

        # add string to label txt
        label_txt += f'{coor_str}\n'
    
    # Write labels to txt file
    with open(f'{path}/{id_}.txt', 'w') as f:
        f.write(label_txt)


def make_tile_data(
    img_file,
    annotations,
    img_size,
    img_path,
    label_path,
    task,
    single_class,
    tile=2
):
    #
    if task == "seg":
        raise ValueError("khong ho tro seg dau")

    img = cv2.imread(img_file)
    img_id = os.path.splitext(os.path.basename(img_file))[0]
    grids = make_grid(512, 512, tile)
    for i, (x1,x2,y1,y2) in enumerate(grids):
        img_cut = img[y1:y2, x1:x2]
        cut_h, cut_w = img_cut.shape[:2]
        label_txt = ''
        for (coordinate, class_name) in annotations:
            # A single mask
            poly = Polygon(coordinate[0])
            mask = rasterio.features.rasterize([poly], out_shape=[512, 512])
            mask_cut = mask[y1:y2, x1:x2]
            # To box
            ys, xs = np.where(mask_cut)
            if len(xs) and len(ys):
                if single_class:
                    label_txt += '0 '
                else:
                    label_txt += f'{class_map[class_name]} '
                # x1, y1, x2, y2
                bx1, bx2 = min(xs), max(xs)
                by1, by2 = min(ys), max(ys)
                bw, bh = bx2 - bx1, by2 - by1
                bxc, byc = bx1 + bw / 2, by1 + bh / 2
                label_txt += f'{bxc/cut_w} {byc/cut_h} {bw/cut_w} {bh/cut_h}\n'
        
        if label_txt:
            cv2.imwrite(os.path.join(img_path, f"{img_id}_{i}.tif"), img_cut)
            with open(f'{label_path}/{img_id}_{i}.txt', 'w') as f:
                f.write(label_txt)


def create_yolo_dataset(
    img_dir="auto",
    ann_path="auto",
    out="datasets",
    nonsingle_class=False,
    fold=0,
    task="det",
    classes=["blood_vessel"],
    data_csv="data.csv",
    tile=2
    
):
    if img_dir == "auto":
        img_dir = auto_img_dir()
    if ann_path == "auto":
        ann_path = auto_ann_path()

    assert os.path.isfile(ann_path), f"{ann_path} not found"

    if os.path.exists(out):
        shutil.rmtree(out)

    train_path = os.path.join(out, "train")
    val_path = os.path.join(out, "val")

    (
        train_image_path, train_label_path,
        val_image_path, val_label_path
    ) = mkdir_yolo_data(train_path, val_path)
        
    # Prepare dataset for yolo training
    annotations_dict = create_vessel_annotations(
        ann_path,  classes
    )

    df = pd.read_csv(data_csv)

    train_ids = set(df[df['fold'] != fold]['id'].tolist())
    val_ids = set(df[df['fold'] == fold]['id'].tolist())

    files = glob(os.path.join(img_dir, "*"))
    img_size = 512

    for img_file in tqdm(files):
        id_ = os.path.splitext(os.path.basename(img_file))[0]

        if tile:
            if id_ in train_ids:
                make_tile_data(
                    img_file,
                    annotations_dict[id_],
                    img_size,
                    train_image_path,
                    train_label_path,
                    task=task,
                    single_class=not nonsingle_class,
                    tile=tile,
                )
            elif id_ in val_ids:
                make_tile_data(
                    img_file,
                    annotations_dict[id_],
                    img_size,
                    val_image_path,
                    val_label_path,
                    task=task,
                    single_class=not nonsingle_class,
                    tile=tile
                )
        else:
            if id_ in train_ids:
                # create label txt file
                create_label_file(id_, annotations_dict[id_],
                    train_label_path, img_size, single_class=not nonsingle_class,
                    task=task)
                shutil.copy2(img_file, train_image_path)
            elif id_ in val_ids:
                create_label_file(id_, annotations_dict[id_],
                    val_label_path, img_size, single_class=not nonsingle_class,
                    task=task)
                shutil.copy2(img_file, val_image_path)

    print(f"Saved to {out}")
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Yolov8 data preparation")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--out", default="datasets")
    parser.add_argument("--ann_path",
        default="auto",
        help="path to polygons.jsonl"
    )
    parser.add_argument("--img_dir",
        default="auto"
    )
    parser.add_argument("--data_csv",
        default="data.csv",
        help="Path to data.csv split file",
    )
    parser.add_argument("--task",
        default="seg",
        help="det or seg",
    )
    parser.add_argument("--classes",
        nargs="+",
        default=["blood_vessel", "unsure"],
        help="det or seg",
    )
    parser.add_argument('--nonsingle_class', action='store_true')
    parser.add_argument("--tile", type=int, default=0, help="Tile image into parts")

    args = parser.parse_args()

    create_yolo_dataset(**vars(args))
