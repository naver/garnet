"""
GaRNet
Copyright (c) 2022-present NAVER Corp.
Apache License v2.0
"""

import torch


class wrap_module:
    def __init__(self, model_type="GaRNet", gpu=True, att_visualization=False):
        self.type = model_type
        self.att_visualization = att_visualization
        if gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

        if self.type == "EnsNet":
            from EnsNet.model import STE

            self.net = STE(3)

        elif self.type == "EraseNet":
            from EraseNet.models.sa_gan import STRnet2

            self.net = STRnet2(3)

        elif self.type == "MTRNet++":
            from MTRNetPlusPlus.src.networks import MaskInpaintGenerator_v5

            self.net = MaskInpaintGenerator_v5(in_channels=4, use_spectral_norm=False)

        elif self.type == "MTRNet":
            from MTRNet.model import MTRNet

            self.net = MTRNet()

        elif self.type == "GaRNet":
            from GaRNet.model import GaRNet

            self.net = GaRNet(3)

    def load(self, state_dict):
        try:
            self.net.load_state_dict(state_dict["model_state_dict"])
        except:
            self.net.load_state_dict(state_dict)
        self.net.to(self.device)
        self.net.eval()

    def normalize_input(self, x, gt):
        if self.type == "MTRNet++" or self.type == "EraseNet":
            x[:, :3], gt = x[:, :3] / 255.0, gt / 255.0
        else:
            x[:, :3], gt = x[:, :3] / 127.5 - 1, gt / 127.5 - 1
        return x, gt

    def postprocess(self, x, gt):
        if self.type == "MTRNet++" or self.type == "EraseNet":
            x, gt = torch.clamp(x, 0, 1) * 255.0, gt * 255.0
        else:
            x, gt = torch.clamp((x + 1), 0, 2) * 127.5, torch.clamp((gt + 1), 0, 2) * 127.5
        return x, gt

    def __call__(self, x):
        if self.type == "EnsNet":
            _, _, _, _, result = self.net(x[:, :3])
        elif self.type == "MTRNet":
            result = self.net(x)
        elif self.type == "EraseNet":
            _, _, _, result, _ = self.net(x[:, :3])
        elif self.type == "MTRNet++":
            result, _, predicted_mask = self.net(x)
            result = result * predicted_mask + x[:, :3] * (1 - predicted_mask)
        elif self.type == "GaRNet":
            _, _, _, _, result, _, att_set, att = self.net(x)
            if self.att_visualization:
                return result, att_set, att
        return result
