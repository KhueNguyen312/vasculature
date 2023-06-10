import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

MINOVERLAP = 0.6 # default value

parser = argparse.ArgumentParser()
parser.add_argument('--oof_file', help="path to oof json file", type=str)
parser.add_argument('--out', type=str, default="ensemble.json", )
args = parser.parse_args()

