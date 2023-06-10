import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math
import numpy as np
from shapely.geometry.polygon import Polygon

MINOVERLAP = 0.6

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, help="Output path", default="metric.txt")
    parser.add_argument('--gt', type=str,
                        default="/data-storage/hubmap_data/fold0/annotations_valid.json",
                        help="Path to ground truth file")
    parser.add_argument('--input', type=str, help="Path to prediction file")

    args = parser.parse_args()
    return args
    
def _build_ann(ann):
    seg = ann["segmentation"]
    seg_arr = np.array(seg).astype(int).reshape(-1, 2)
    try:
        poly = Polygon(seg_arr)
    except:
        print(seg_arr)
        raise
    return {
        "bbox": poly.bounds,
        "seg": seg,
        "image_id_": ann['image_id_'],
        "class_name": "blood_vessel",
        "used": False
    }

def load_coco_predictions(path):
    res = []
    with open(path, "r") as f:
        anns = json.load(f)
        if isinstance(anns, dict):
            anns = anns['annotations']

    for ann in anns:
       res.append(_build_ann(ann))

    return res

def load_coco_annotations(path):
    # Map by file
    res = {}
    with open(path, "r") as f:
        anns = json.load(f)
        if isinstance(anns, dict):
            anns = anns['annotations']

    for ann in anns:
       if ann['image_id_'] in res:
           res[ann['image_id_']].append(_build_ann(ann))
       else:
           res[ann['image_id_']] = [_build_ann(ann)]
    
    return res

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

def run_eval(gt_anns, pred_anns, output_file):
    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    gt_counter_per_class = {'blood_vessel': sum(len(x) for x in gt_anns.values())}
    counter_images_per_class = {}
    with open(output_file, "w") as f:
        f.write("# AP and precision/recall per class\n")
        gt_classes = ['blood_vessel']
        class_name = 'blood_vessel'
        
        nd = len(pred_anns)
        tp = [0] * nd # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(pred_anns):
            image_id = detection['image_id_']
            if image_id not in gt_anns:
                raise ValueError(f"{image_id} not in {list(gt_anns.keys())},{len(gt_anns.keys())}")
            ground_truth_data = gt_anns[image_id]
            ovmax = -1
            gt_match = -1
            epsilon = 1e-6
            # load detected object bounding-box
            bb = detection["bbox"]
            for obj in ground_truth_data:
                # look for a class_name match
                if obj["class_name"] == class_name:
                    bbgt = obj["bbox"]
                    bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                    iw = bi[2] - bi[0] + epsilon
                    ih = bi[3] - bi[1] + epsilon
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + epsilon) * (bb[3] - bb[1] + epsilon) + (bbgt[2] - bbgt[0]
                                        + epsilon) * (bbgt[3] - bbgt[1] + epsilon) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            # assign detection as true positive/don't care/false positive
            # set minimum overlap
            min_overlap = MINOVERLAP
            if ovmax >= min_overlap:
                if not bool(gt_match["used"]):
                    # true positive
                    tp[idx] = 1
                    gt_match["used"] = True

                else:
                    # false positive (multiple detection)
                    fp[idx] = 1
            else:
                # false positive
                fp[idx] = 1
                if ovmax > 0:
                    status = "INSUFFICIENT OVERLAP"

        #print(tp)
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        #print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx] + 1e-6)
        # print(prec)
        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)
        """
            Write to output.txt
        """
        rounded_prec = [ '%.2f' % elem for elem in prec ]
        rounded_rec = [ '%.2f' % elem for elem in rec ]
        f.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
        print(text)
        ap_dictionary[class_name] = ap

        # n_images = counter_images_per_class[class_name]
        # lamr, mr, fppi = log_average_miss_rate(np.array(prec), np.array(rec), n_images)
        # lamr_dictionary[class_name] = lamr

        f.write("\n# mAP of all classes\n")
        mAP = sum_AP / 1
        text = "mAP = {0:.2f}%".format(mAP*100)
        print(text)

if __name__ == '__main__':
    args = parse_args()
    
    gt_anns = load_coco_annotations(args.gt)
    pred_anns = load_coco_predictions(args.input)

    print(len(pred_anns))
    run_eval(gt_anns, pred_anns, args.out)
