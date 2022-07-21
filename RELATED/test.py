"""
GaRNet
Copyright (c) 2022-present NAVER Corp.
Apache License v2.0
"""

import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import TestDataset, MyDataset
from wrap_module import wrap_module


parser = argparse.ArgumentParser(description="Hyperparams")
parser.add_argument("--result_path", nargs="?", type=str, default="./result", help="result image path")
parser.add_argument("--test_path", nargs="?", type=str, default="./data", help="the path of test image")
parser.add_argument("--use_json", action="store_true", help="Use json file")
parser.add_argument(
    "--box_path",
    nargs="?",
    type=str,
    default="./data",
    help="the path of box .txt file. When use_json is true, this path is ignored",
)
parser.add_argument("--model_type", nargs="?", type=str, default="EnsNet")
parser.add_argument("--model_path", nargs="?", type=str, default="../WEIGHTS/EnsNet/saved_model.pth")
parser.add_argument("--batch", nargs="?", type=int, default=5)
parser.add_argument("--input_size", nargs="?", type=int, default=512)
parser.add_argument("--use_composited", action="store_true", help="Use composited Image")
parser.add_argument("--gpu", action="store_true", help="Use gpu")
args = parser.parse_args()


def test(args, net, device="cpu"):

    # ============================load data===========================

    test_path = args.test_path

    if args.use_json:
        test_data = MyDataset(test_path, input_size=args.input_size, normalize=False)
    else:
        test_data = TestDataset(test_path, args.box_path, input_size=args.input_size, normalize=False)

    test_loader = DataLoader(test_data, batch_size=args.batch, shuffle=False)

    for batch_idx, (original, mask, _, name) in enumerate(test_loader):
        x = torch.cat([original, mask], dim=1)
        x, _ = net.normalize_input(x, x)
        x = x.to(device)

        print(batch_idx)

        result = net(x)

        if args.use_composited:
            result = (1 - mask) * x[:, :3].cpu() + mask * result.cpu()

        result, _ = net.postprocess(result, result)

        for idx, img in enumerate(result.cpu()):

            img = ((img.cpu().detach().numpy().transpose(1, 2, 0))).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.result_path, name[idx]), img)

    print("complete")


if __name__ == "__main__":

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    if torch.cuda.is_available() and args.gpu:
        device = "cuda"
        use_gpu = True
    else:
        device = "cpu"
        use_gpu = False

    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True
    # device = "cpu"
    print("running on %s" % device)

    net = wrap_module(args.model_type, gpu=use_gpu)
    net.load(torch.load(args.model_path, map_location=device))
    test(args, net, device)
