"""
GaRNet
Copyright (c) 2022-present NAVER Corp.
Apache License v2.0
"""

import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info

from dataloader import MyDataset
from utils import calc_PSNR, calc_AGE, ssim
from model import GaRNet


parser = argparse.ArgumentParser(description="Hyperparams")
parser.add_argument("--model_path", nargs="?", type=str, default="../../../2022_ECCV_Supplementary/WEIGHTS/GaRNet/saved_model.pth")
parser.add_argument("--test_path", nargs="?", type=str, default="../DATA/JSON/REAL/test.json")
parser.add_argument("--batch", nargs="?", type=int, default=10)
parser.add_argument("--gpu", action="store_true", help="Use gpu")

args = parser.parse_args()


def evaluate(args, net):
    batch_size = args.batch

    device = "cuda" if args.gpu else "cpu"

    # ============================load data===========================

    test_data = MyDataset(args.test_path, input_size=512)

    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=5, shuffle=False, drop_last=False)
    test_size = len(test_data)
    print("test data : %d" % (test_size))

    evaluation_metric = {"PSNR": 0, "SSIM": 0, "PSNR_C": 0, "SSIM_C": 0, "AGE": 0, "AGE_C": 0}

    with torch.no_grad():
        for batch_idx, (original, ground_truth, _, _) in enumerate(test_dataloader):
            x = original.to(device)
            box_mask = original[:, 3].unsqueeze(1)

            _, _, _, _, result, _, _, _ = net(x)
            result = result.cpu().detach()
            result = ground_truth[:, :3] * (1 - box_mask) + result * box_mask
            result_img = torch.clamp(result + 1, 0, 2) * 127.5
            gt_img = torch.clamp(ground_truth + 1, 0, 2) * 127.5

            img_gray = (
                0.299 * result_img[:, 0, :, :] + 0.587 * result_img[:, 1, :, :] + 0.114 * result_img[:, 2, :, :]
            ).unsqueeze(1)
            gt_gray = (0.299 * gt_img[:, 0, :, :] + 0.587 * gt_img[:, 1, :, :] + 0.114 * gt_img[:, 2, :, :]).unsqueeze(
                1
            )

            evaluation_metric["SSIM"] += ssim(img_gray, gt_gray, data_range=255.0, reduction="sum")
            evaluation_metric["PSNR"] += calc_PSNR(result_img, gt_img).sum()
            evaluation_metric["AGE"] += calc_AGE(img_gray, gt_gray).sum()

            print("%d/%d" % (batch_idx, len(test_data) // batch_size))

        print(
            "PSNR: %f, SSIM: %f, AGE: %f"
            % (
                evaluation_metric["PSNR"] / test_size,
                evaluation_metric["SSIM"] / test_size,
                evaluation_metric["AGE"] / test_size,
            )
        )
        print("complete")


def calc_speed(net):
    macs, params = get_model_complexity_info(
        net, (4, 512, 512), as_strings=True, print_per_layer_stat=False, verbose=False
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))

    dummy = torch.randn(1, 4, 512, 512, dtype=torch.float).to(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    mean_inference_time = 0
    total_iteration = 3000
    
    for _ in range(10):
        _ = net(dummy)

    with torch.no_grad():
        for _ in range(total_iteration):
            start.record()
            _ = net(dummy)
            end.record()
            torch.cuda.synchronize()
            mean_inference_time += start.elapsed_time(end)

    return mean_inference_time / total_iteration


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True

    print("running on %s" % device)

    net = GaRNet(3)

    try:
        net.load_state_dict(torch.load(args.model_path, map_location=device)["model_state_dict"])
    except:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.to(device)
    #evaluate(args, net)
    mGPUTime = calc_speed(net)
    print("mGPUTime: %f" % (mGPUTime))
