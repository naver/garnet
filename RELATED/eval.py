"""
GaRNet
Copyright (c) 2022-present NAVER Corp.
Apache License v2.0
"""

import argparse

import torch
import numpy as np
from dataloader import MyDataset
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info

from utils import calc_PSNR, calc_AGE, ssim
from wrap_module import wrap_module


parser = argparse.ArgumentParser(description="Hyperparams")
parser.add_argument("--model_type", nargs="?", type=str, default="EnsNet")
parser.add_argument("--model_path", nargs="?", type=str, default="../WEIGHTS/EnsNet/saved_model.pth")
parser.add_argument("--test_path", nargs="?", type=str, default="../DATA/JSON/REAL/test.json")
parser.add_argument("--input_size", nargs="?", type=int, default=512)
parser.add_argument("--batch", nargs="?", type=int, default=10)
parser.add_argument("--gpu", action="store_true", help="Use gpu")
args = parser.parse_args()


def evaluate(args, net, device="cpu"):
    batch_size = args.batch

    # ============================load data===========================

    test_data = MyDataset(args.test_path, input_size=args.input_size, normalize=False)

    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=20, shuffle=False, drop_last=False)
    test_size = len(test_data)
    print("test data : %d" % (test_size))

    evaluation_metric = {"PSNR": 0, "SSIM": 0, "PSNR_C": 0, "SSIM_C": 0, "AGE": 0, "AGE_C": 0}

    with torch.no_grad():
        for batch_idx, (original, mask, ground_truth, _) in enumerate(test_dataloader):
            x = torch.cat([original, mask], dim=1)
            x, ground_truth = net.normalize_input(x, ground_truth)

            x = x.to(device)
            box_mask = mask

            result = net(x)
            result = result.cpu().detach()
            result_img, gt_img = net.postprocess(result, ground_truth)

            img_gray = (
                0.299 * result_img[:, 0, :, :] + 0.587 * result_img[:, 1, :, :] + 0.114 * result_img[:, 2, :, :]
            ).unsqueeze(1)
            gt_gray = (0.299 * gt_img[:, 0, :, :] + 0.587 * gt_img[:, 1, :, :] + 0.114 * gt_img[:, 2, :, :]).unsqueeze(
                1
            )

            evaluation_metric["SSIM"] += ssim(img_gray, gt_gray, data_range=255.0, reduction="sum")
            evaluation_metric["PSNR"] += calc_PSNR(result_img, gt_img).sum()
            evaluation_metric["AGE"] += calc_AGE(img_gray, gt_gray).sum()

            result_img = gt_img * (1 - box_mask) + result_img * (box_mask)
            img_gray = gt_gray * (1 - box_mask) + img_gray * (box_mask)

            evaluation_metric["SSIM_C"] += ssim(img_gray, gt_gray, data_range=255.0, reduction="sum")
            evaluation_metric["PSNR_C"] += calc_PSNR(result_img, gt_img).sum()
            evaluation_metric["AGE_C"] += calc_AGE(img_gray, gt_gray).sum()
            result = ground_truth[:, :3] * (1 - box_mask) + result * box_mask

            print("%d/%d" % (batch_idx, len(test_data) // batch_size))

        print(
            "PSNR: %f/%f , SSIM: %f/%f, AGE: %f/%f"
            % (
                evaluation_metric["PSNR"] / test_size,
                evaluation_metric["PSNR_C"] / test_size,
                evaluation_metric["SSIM"] / test_size,
                evaluation_metric["SSIM_C"] / test_size,
                evaluation_metric["AGE"] / test_size,
                evaluation_metric["AGE_C"] / test_size,
            )
        )

        print("complete")


def calc_speed(net):
    if args.model_type == "EnsNet" or args.model_type == "EraseNet":
        shape = (1, 3, args.input_size, args.input_size)
    else:
        shape = (1, 4, args.input_size, args.input_size)
    macs, params = get_model_complexity_info(net, shape[1:], as_strings=True, print_per_layer_stat=False, verbose=False)
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

    if torch.cuda.is_available() and args.gpu:
        device = "cuda"
        use_gpu = True
    else:
        device = "cpu"
        use_gpu = False
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True
    print("running on %s" % device)

    net = wrap_module(args.model_type, gpu=use_gpu)

    net.load(torch.load(args.model_path, map_location=device))
    evaluate(args, net, device)
    mGPUTime = calc_speed(net.net)
    print("mGPUTime: %f" % (mGPUTime))
