"""
GaRNet
Copyright (c) 2022-present NAVER Corp.
Apache License v2.0
"""

import argparse
import os
import glob

import torch
import torchvision
import numpy as np
import cv2

from model import GaRNet


parser = argparse.ArgumentParser(description="Hyperparams")
parser.add_argument("--result_path", nargs="?", type=str, default="./result", help="result image path")
parser.add_argument("--image_path", nargs="?", type=str, default="../DATA/EXAMPLE/IMG", help="test image path")
parser.add_argument("--box_path", nargs="?", type=str, default="../DATA/EXAMPLE/TXT")
parser.add_argument("--input_size", nargs="?", type=int, default="512")
parser.add_argument("--model_path", nargs="?", type=str, default="../WEIGHTS/GaRNet/saved_model.pth")
parser.add_argument("--attention_vis", action="store_true", help="Visualize Attention score map")
parser.add_argument("--gpu", action="store_true", help="Use gpu")

args = parser.parse_args()


def attention_grid_interpolation(im, att):
    pad = torch.nn.ConstantPad2d(0, 1)
    im = pad(im)
    im = ((im.cpu().detach().numpy().transpose(0, 2, 3, 1) + 1).clip(0, 2) * 127.5).astype(np.uint8)
    im = im.reshape(1, im.shape[0] * im.shape[1], im.shape[2], im.shape[3])[0]

    att = pad(att).cpu().detach().numpy()
    att = att[0][0]
    opacity = att
    opacity = opacity[..., np.newaxis]
    opacity = opacity * 0.95 + 0.05
    heatmap = (opacity * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(255 - heatmap, cv2.COLORMAP_JET)

    vis_im = cv2.addWeighted(im, 0.6, heatmap, 0.4, 0)

    return torch.from_numpy(vis_im.transpose(2, 0, 1))


def get_box(box_path, original_size, input_size, dilation=0):

    box_mask = np.expand_dims(np.zeros(original_size), axis=2).astype(np.uint8)
    with open(box_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split("##")[0]
            try:
                line = list(map(int, line.split(" ")))
            except:
                line = list(map(int, line.split(",")))
            point = np.array([[line[i], line[i + 1]] for i in range(0, len(line), 2)], np.int32)
            box_mask = cv2.fillPoly(box_mask, [point], 1)

    kernel = np.ones((3, 3), np.uint8)
    box_mask = cv2.dilate(box_mask, kernel, iterations=dilation)
    box_mask = box_mask.astype(np.float32)
    box_mask = np.expand_dims(
        cv2.resize(box_mask, (input_size, input_size), interpolation=cv2.INTER_NEAREST), axis=0
    ).astype(np.float32)

    return box_mask


def test(args, net):

    device = "cuda" if args.gpu else "cpu"

    # ============================load data===========================

    image_list = glob.glob(os.path.join(args.image_path, "*.jpg"))
    box_list = glob.glob(os.path.join(args.box_path, "*.txt"))
    image_list.sort()
    box_list.sort()
    count = 0
    for batch_idx, (im_path, box_path) in enumerate(zip(image_list, box_list)):

        # prepare input
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        name = im_path.rsplit("/", 1)[1]
        H, W, _ = im.shape

        box_mask = get_box(box_path, (H, W), args.input_size)

        # resize and normalization
        im = cv2.resize(im, (args.input_size, args.input_size)).transpose(2, 0, 1).astype(np.float32)
        im = im / 127.5 - 1
        im, box_mask = torch.FloatTensor(im), torch.FloatTensor(box_mask)

        x = torch.cat([im, box_mask], axis=0).unsqueeze(0).to(device)
        print(batch_idx)

        # inference
        with torch.no_grad():
            _, _, _, _, result, _, att_set, att = net(x)

        # save result
        result = (1 - box_mask) * im + box_mask * result.cpu()
        img = (torch.clamp(result[0] + 1, 0, 2) * 127.5).cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (H, W))
        cv2.imwrite(os.path.join(args.result_path, name), img)

        if args.attention_vis:
            stroke_att = []
            surround_att = []
            gate_att = []
            for i in range(0, 4):
                stroke_att_map = attention_visualize(result.cpu(), att_set[i][0], args.input_size)
                surround_att_map = attention_visualize(result.cpu(), att_set[i][1], args.input_size)
                gate_att_map = attention_visualize(
                    result.cpu(), (att[i] - att[i].min()) / (att[i].max() - att[i].min()), args.input_size
                )
                stroke_att.append(stroke_att_map)
                surround_att.append(surround_att_map)
                gate_att.append(gate_att_map)
            stroke_att_map = torchvision.utils.make_grid(stroke_att, nrow=4, padding=10, pad_value=255).permute(1, 2, 0)
            surround_att_map = torchvision.utils.make_grid(surround_att, nrow=4, padding=10, pad_value=255).permute(
                1, 2, 0
            )
            gate_att_map = torchvision.utils.make_grid(gate_att, nrow=4, padding=10, pad_value=255).permute(1, 2, 0)
            stroke_att_map = cv2.cvtColor(stroke_att_map.numpy(), cv2.COLOR_RGB2BGR)
            surround_att_map = cv2.cvtColor(surround_att_map.numpy(), cv2.COLOR_RGB2BGR)
            gate_att_map = cv2.cvtColor(gate_att_map.numpy(), cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(args.result_path, name.rsplit(".", 1)[0] + "_stroke_attention.jpg"), stroke_att_map
            )
            cv2.imwrite(
                os.path.join(args.result_path, name.rsplit(".", 1)[0] + "_surround_attention.jpg"), surround_att_map
            )
            cv2.imwrite(os.path.join(args.result_path, name.rsplit(".", 1)[0] + "_gate_attention.jpg"), gate_att_map)
        count += 1
        if count >= 800:
            break
    print("complete")


def attention_visualize(result, att, input_size):

    att = torch.nn.functional.interpolate(att, size=(input_size, input_size), mode="bilinear", align_corners=True)
    att = attention_grid_interpolation(result, att)

    return att


if __name__ == "__main__":

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
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
    test(args, net)
