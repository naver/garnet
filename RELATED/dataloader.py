"""
GaRNet
Copyright (c) 2022-present NAVER Corp.
Apache License v2.0
"""

import os
import glob
import json

import torch
import numpy as np
import cv2
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, im_path, box_path, input_size=512, normalize=True):
        self.image_list = glob.glob(os.path.join(im_path, "*.jpg"))
        self.box_list = glob.glob(os.path.join(box_path, "*.txt"))
        self.normalize = normalize

        self.image_list.sort()
        print(len(self.image_list))
        self.box_list.sort()
        self.input_size = input_size

    def __getitem__(self, index):
        image_path = self.image_list[index]
        box_path = self.box_list[index]
        x = cv2.imread(image_path)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        box_mask = np.expand_dims(np.zeros(x.shape[:-1]), axis=2).astype(np.uint8)

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
        box_mask = box_mask.astype(np.uint8)

        x = (cv2.resize(x, (self.input_size, self.input_size))).transpose(2, 0, 1)
        box_mask = (cv2.resize(box_mask, (self.input_size, self.input_size))).transpose(2, 0, 1)
        box_mask = box_mask.astype(np.bool).astype(np.float32)

        if self.normalize:
            x = (x / 127.5) - 1
        return (
            torch.cat(torch.from_numpy(x.astype(np.float32)), box_mask, dim=0),
            np.array(0),
            box_mask,
            image_path.rsplit("/", 1)[1],
        )

    def __len__(self):
        return len(self.image_list)


class MyDataset(Dataset):
    def __init__(self, path, input_size=512, augmentation=False, normalize=True):
        with open(path, "r") as f:
            self.data = json.load(f)
        self.normalize = normalize
        self.size = input_size
        self.mask_threshold = 25
        self.augmentation = augmentation
        self.backup = self.data[0]

    def __getitem__(self, index):
        image_path = self.data[index]["dir"]
        gt_path = self.data[index]["gt_dir"]
        box = self.data[index]["word_bb"]

        x = cv2.imread(image_path)
        y = cv2.imread(gt_path)

        try:
            x, y = cv2.cvtColor(x, cv2.COLOR_BGR2RGB), cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
            self.backup = self.data[index]
        except:
            image_path = self.backup["dir"]
            gt_path = self.backup["gt_dir"]
            box = self.backup["word_bb"]
            x = cv2.imread(image_path)
            y = cv2.imread(gt_path)
            x, y = cv2.cvtColor(x, cv2.COLOR_BGR2RGB), cv2.cvtColor(y, cv2.COLOR_BGR2RGB)

        if x.shape != y.shape:
            y = cv2.resize(y, (x.shape[1], x.shape[0]))

        box_mask = self.get_box_mask(x, self.size, box)

        x, y = self.resize(x, self.size), self.resize(y, self.size)

        if box_mask.ndim == 2:
            box_mask = np.expand_dims(box_mask, axis=2).astype(np.float32)

        y = y * box_mask + x * (1 - box_mask)
        mask = np.greater(
            np.mean(np.abs(np.array(x).astype(np.float32) - np.array(y).astype(np.float32)), axis=-1),
            self.mask_threshold,
        ).astype(np.uint8)
        mask = np.expand_dims(mask, axis=0).astype(np.float32)

        x, y = torch.from_numpy(x.transpose(2, 0, 1)).type(torch.FloatTensor), torch.from_numpy(
            y.transpose(2, 0, 1)
        ).type(torch.FloatTensor)
        if self.normalize:
            x, y = (x / 127.5) - 1, (y / 127.5) - 1

        return x, torch.FloatTensor(box_mask.transpose(2, 0, 1)), y, image_path.rsplit("/", 1)[1]

    def resize(self, img, size, crop=True):
        img_h, img_w = img.shape[0:2]

        if crop and img_h != img_w:
            shortest_length = min(img_h, img_w)
            offset_x = (img_w - shortest_length) // 2
            offset_y = (img_h - shortest_length) // 2
            img = img[offset_y : offset_y + shortest_length, offset_x : offset_x + shortest_length]

        img = cv2.resize(img, (size, size))

        return img

    def get_box_mask(self, img, size, boxes, crop=True):
        box_mask = np.expand_dims(np.zeros(img.shape[:-1]), axis=2).astype(np.uint8)

        for idx, box in enumerate(boxes):
            point = np.array([[box[i], box[i + 1]] for i in range(0, len(box), 2)], np.int32)
            box_mask = cv2.fillPoly(box_mask, [point], 1).astype(np.float32)

        img_h, img_w = box_mask.shape[0:2]

        if crop and img_h != img_w:
            shortest_length = min(img_h, img_w)
            offset_x = (img_w - shortest_length) // 2
            offset_y = (img_h - shortest_length) // 2
            box_mask = box_mask[offset_y : offset_y + shortest_length, offset_x : offset_x + shortest_length]

        box_mask = np.expand_dims(cv2.resize(box_mask, (size, size), interpolation=cv2.INTER_NEAREST), axis=2).astype(
            np.float32
        )
        return box_mask

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    train_data = MyDataset("test", 512)
    # train_data = TestDataset(path="/DATA/SCUT/test")
    print("train data %d " % len(train_data))
    print("image shape")
