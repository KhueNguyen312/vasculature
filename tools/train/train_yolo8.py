import sys
sys.path.append(f"./ultralytics")
sys.path.append(f"./tools")
import os
from ultralytics import YOLO
import torch
from comp2yolo8 import create_yolo_dataset
from tool_utils.model import infer_yolo8
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--data", default="datasets", type=str)
    parser.add_argument("--task", type=str, default="det", help="det or seg")
    parser.add_argument("--weight", type=str, default="yolov8l.pt")
    parser.add_argument("--project", type=str, default="../runs")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--skip_data", action="store_true", help="Skip re-create dataset")
    parser.add_argument("--train_pseudo", action="store_true")
    return parser.parse_args()


# ==================== MAIN ==================== #

args = parse_args()

args.data += f"_f{args.fold}"

if args.tile:
    args.data += f"_tile{args.tile}"

if not args.skip_data or not os.path.isdir(args.data):
    dataset_path = create_yolo_dataset(
        out=args.data,
        fold=args.fold,
        task=args.task,
        classes=["blood_vessel"],
        tile=args.tile
    )
else:
    dataset_path =  args.data

# Edit yaml content
yaml_content = f'''
path: {dataset_path}
train: train/images
val: val/images

names:
  0: blood_vessel
'''

yaml_file = 'data.yaml'
cfg_file ='cfg.yaml'

with open(yaml_file, 'w') as f:
    f.write(yaml_content)

cfg = f'''
# copy_paste: 0.25
mixup: 0.5
mosaic: 0.5
flipud: 0.5
degrees: 15
'''

with open(cfg_file, 'w') as f:
    f.write(cfg)

# training
model = YOLO(args.weight)

model_name = args.weight.replace(".pt", "").replace("-seg", "")
args.name = f"{args.name}_{model_name }_{args.task}_{args.imgsz}_f{args.fold}"

if args.tile:
    args.name += f"_tile{args.tile}"
results = model.train(
    batch=args.batch,
    device=args.device,
    data=yaml_file,
    epochs=args.epochs,
    imgsz=args.imgsz,
    workers=4,
    cfg=cfg_file,
    project=args.project,
    name=args.name, # Name of experiment
    close_mosaic=0#int(args.epochs * 0.15),
)

infer_yolo8(
    model, f"{dataset_path}/val/images",
    os.path.join(args.project, args.name, f"oof_fold{args.fold}.json")
)

if args.train_pseudo:
    # Merge pseudo label to current yolo bboxes
    