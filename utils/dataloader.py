import pandas as pd
import numpy as np
import torch
import torchvision
import imageio
import os
import glob
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL.ImageDraw import Draw
import PIL
from PIL import Image
from bounding_box import bounding_box as bb
from albumentations.augmentations import transforms
import cv2
from torch import nn
from albumentations import (
    ShiftScaleRotate,
    Compose,
    GridDistortion,
    OpticalDistortion,
    Blur,
    GaussNoise,
    RandomBrightness,
    RandomContrast,
    RandomGamma,
    OneOf,
    Resize,
)


class2int = {}
int2class = {}
root = "/home/ubuntu/Documents/dataset/"

# Make dictionary
csv = pd.read_csv(os.path.join(root, "sample_data/sample_data.csv"))
for i, name in enumerate(sorted(np.unique(csv["class"]))):
    class2int[name] = i
    int2class[i] = name


class Dataset(object):
    def __init__(self, root, transforms, train=True):
        """
        root: root path, str
        transforms: augmentation function
        select_class: to select class for training model. list
        """

        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(glob.glob(os.path.join(root, "sample_data/*/*"))))
        self.labels = pd.read_csv(os.path.join(root, "sample_data/sample_data.csv"))
        self.train = train

        self.select_class = [
            "bollard",
            "bus",
            "car",
            "motorcycle",
            "movable_signage",
            "person",
            "pole",
            "potted_plant",
            "traffic_light",
            "traffic_sign",
            "tree_trunk",
            "truck",
            "wheelchair",
        ]
        self._select_class()

    def __getitem__(self, idx):
        # load images ad masks
        img = imageio.imread(self.imgs[idx])
        img = np.float32(img) / img.max()

        labels = self.labels[
            (self.labels["filename"] == self.imgs[idx].split("/sample_data/")[-1])
        ]
        num_objs = labels.shape[0]
        boxes = []

        xmins, xmaxs, ymins, ymaxs, classes = (
            labels["xmin"],
            labels["xmax"],
            labels["ymin"],
            labels["ymax"],
            labels["class"],
        )
        obj_class = []

        for xmin, xmax, ymin, ymax, c in zip(xmins, xmaxs, ymins, ymaxs, classes):
            boxes.append([xmin, ymin, xmax, ymax])
            obj_class.append(class2int[c])

        if self.transforms is not None and self.train:
            img, boxes = self.transforms(img, boxes)

        img = np.transpose(img, (2, 0, 1))
        # convert everything into a torch.Tensor
        img = torch.as_tensor(img, dtype=torch.float32)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(obj_class, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        return img, boxes, labels

    def __len__(self):
        return len(self.imgs)

    def _select_class(self):
        i = 0
        while i < self.labels.shape[0]:
            if self.labels.iloc[i]["class"] in self.select_class:
                i += 1
            else:
                self.labels = self.labels.drop(self.labels.index[i])
        self.imgs = [
            img
            for img in self.imgs
            if img.split("/sample_data/")[-1] in list(self.labels["filename"])
        ]

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return (
            images,
            boxes,
            labels,
        )  # tensor (N, 3, 300, 300), 3 lists of N tensors each


def get_aug(p=0.5):
    return Compose(
        [
            RandomBrightness(limit=0.2, p=p),
            RandomContrast(limit=0.2, p=p),
            OneOf(
                [
                    OpticalDistortion(
                        distort_limit=0.2,
                        shift_limit=0,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=0.2,
                    ),
                    # GridDistortion(distort_limit=0.2, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.2) # bbox도 같이 변화하지 않는다.
                ],
                p=p,
            ),
            Resize(800, 800),
        ]
    )


class Transform:
    def __init__(self, transforms):
        self.transform = transforms

    def __call__(self, image, target):
        h, w, _ = image.shape
        transformed = self.transform(image=image)
        x_ratio, y_ratio = 800 / w, 800 / h
        new_boxes = []
        for b in target:
            new_boxes.append(
                [b[0] * x_ratio, b[1] * y_ratio, b[2] * x_ratio, b[3] * y_ratio]
            )

        return transformed["image"], new_boxes

