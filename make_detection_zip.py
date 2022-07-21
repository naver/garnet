"""
GaRNet
Copyright (c) 2022-present NAVER Corp.
Apache License v2.0
"""

import os
from glob import glob
import argparse
import shutil
import numpy as np

parser = argparse.ArgumentParser(description="Hyperparams")
parser.add_argument(
    "--path", nargs="?", type=str, default="./DATA/REAL/test/all_gts", help="txt file for detection eval"
)
parser.add_argument(
    "--save_path", nargs="?", type=str, default="./DATA/REAL/test/all_gts/tmp", help="txt file for detection eval"
)
parser.add_argument("--is_gt", action="store_true", help="make gt file")
args = parser.parse_args()

file_list = glob(os.path.join(args.path, "*.txt"))

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
file_list.sort()
print(len(file_list))
for idx, file_path in enumerate(file_list):
    if args.is_gt:
        dpath = os.path.join(args.save_path, "gt_img_" + str(idx + 1) + ".txt")
    else:
        dpath = os.path.join(args.save_path, "res_img_" + str(idx + 1) + ".txt")

    with open(file_path, "r") as rf, open(dpath, "w") as df:
        lines = rf.readlines()
        for line in lines:
            box = np.array(list(map(float, line[:-1].split(","))))
            box = box.reshape(-1, 2).transpose(1, 0).astype(np.int32)
            df.write(f"{min(box[0])}, {min(box[1])}, {max(box[0])}, {max(box[1])}")
            if args.is_gt:
                df.write(", ' '")
            df.write("\n")
