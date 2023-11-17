from __future__ import print_function
# region
import os
import numpy as np
import torch
from PIL import Image, ImageFilter
from skimage import color
from torchvision import transforms, datasets

import torch.nn as nn


import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from loguru import logger

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from torchvision import transforms
from torchvision.models import resnet50
import torch.optim as optim
import torch.nn as nn


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models import resnet50
import torch.optim as optim
import torch.nn as nn
import argparse
from collections import defaultdict

import copy
import os
import torch
import numpy as np
import random
from sklearn.model_selection import KFold

import time

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel

import json
import pickle

from loguru import logger

import os
import cv2
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import albumentations as A
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

# endregion
import torch
import torch.nn as nn
import torch.utils.data.distributed
import torch.multiprocessing as mp

import sys
sys.path.append("/home/liilu/Desktop/COURSE/Pic_proc/AIGC/BASELINE/augmentIQ")

from augmentIQ.options.train_options import TrainOptions
from augmentIQ.datasets.iqa_distortions import *
from einops import rearrange, repeat


from augmentIQ.content_aware_feats import ContentAwareFeats
from augmentIQ.quality_aware_feats import QualityAwareFeats
from ImageReward.ImageReward import ImageReward
import scipy.io

import glob
import sys

# region


class AIGC_3K(Dataset):
    def __init__(self, opt, blip_tokenizer, flag="train"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.opt = opt
        if self.opt.aug:
            self.n_args = self.opt.n_args
        else:
            self.n_args = 1

        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform = transform
        self.aug = opt.aug
        self.blip_tokenizer = blip_tokenizer
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.__read_data__()

    def __read_data__(self):
        self.data_frame = pd.read_csv(self.opt.csv_file)
        self.img_dir = self.opt.image_folder

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.data_frame.at[idx, "name"])
        image = Image.open(img_name)

        prompt = self.data_frame.at[idx, "prompt"]
        text_input = self.blip_tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt")
        assert isinstance(prompt, str)

        # adj1 = "" if not isinstance(self.data_frame.at[idx, "adj1"], str) else self.data_frame.at[idx, "adj1"]
        # adj2 = "" if not isinstance(self.data_frame.at[idx, "adj2"], str) else self.data_frame.at[idx, "adj2"]
        # style = "" if not isinstance(self.data_frame.at[idx, "style"], str) else self.data_frame.at[idx, "style"]

        mos_quality = self.data_frame.at[idx, "mos_quality"]
        std_quality = self.data_frame.at[idx, "std_quality"]
        mos_align = self.data_frame.at[idx, "mos_align"]
        std_align = self.data_frame.at[idx, "std_align"]
        assert isinstance(mos_quality, float) and isinstance(std_quality, float) and isinstance(mos_align, float) and isinstance(std_align, float)

        sample = dict()

        if self.aug:
            n_args = self.opt.n_args
            aug_images = self._augment_images(image, n_args)
            img_list = aug_images
        else:
            n_args = 1
            img_list = [image]

        aug_img_tensors = []
        if self.transform:
            for i in range(len(img_list)):
                img_list[i] = self.transform(img_list[i])
                aug_img_tensors.append(img_list[i])

        aug_img_tensors = torch.stack(aug_img_tensors, dim=0)
        # logger.info(f"aug_img_tensors.shape:{aug_img_tensors.shape}")

        sample['image'] = aug_img_tensors  # D X C X H X W

        sample['mos_quality'] = np.tile(mos_quality, self.n_args)  # D
        sample['std_quality'] = np.tile(std_quality, self.n_args)  # D
        sample['mos_align'] = np.tile(mos_align, self.n_args)  # D
        sample['std_align'] = np.tile(std_align, self.n_args)  # D
        sample['prompt_input_ids'] = text_input.input_ids.expand(self.n_args, -1)  # D X _specify_length_
        sample['prompt_attention_mask'] = text_input.attention_mask.expand(self.n_args, -1)  # D X _specify_length_

        return sample

    def _augment_images(self, img, n_args=40):
        augmentation_lists = []
        for level in range(0, 4):
            # Define the transforms
            augmentation_level_list = [
                lambda x: imjitter(x, level),
                lambda x: imblurgauss(x, level),
                lambda x: imblurlens(x, level),
                lambda x: imblurmotion(x, level),
                lambda x: imcolordiffuse(x, level),
                lambda x: imcolorshift(x, level),
                lambda x: imcolorsaturate(x, level),
                lambda x: imsaturate(x, level),
                lambda x: imcompressjpeg(x, level),
                lambda x: imnoisegauss(x, level),
                lambda x: imnoisecolormap(x, level),
                lambda x: imnoiseimpulse(x, level),
                lambda x: imnoisemultiplicative(x, level),
                lambda x: imdenoise(x, level),
                lambda x: imbrighten(x, level),
                lambda x: imdarken(x, level),
                lambda x: immeanshift(x, level),
                lambda x: imresizedist(x, level),
                lambda x: imresizedist_bilinear(x, level),
                lambda x: imresizedist_nearest(x, level),
                lambda x: imresizedist_lanczos(x, level),
                lambda x: imsharpenHi(x, level),
                lambda x: imcontrastc(x, level),
                lambda x: imcolorblock(x, level),
                lambda x: impixelate(x, level),
                lambda x: imnoneccentricity(x, level),
                # lambda x: imwarpmap(x, np.random.randn(x.size[0], x.size[1], 2)),
                lambda x: imjitter(x, level),
            ]
            augmentation_lists += augmentation_level_list

        # Randomly select n_args augmentation methods from the list
        selected_augmentations = random.sample(augmentation_lists, n_args)

        # Apply the selected augmentations to the input image
        augmented_images = []
        for augmentation in selected_augmentations:
            augmented_image = augmentation(img)
            augmented_images.append(augmented_image)

        return augmented_images


def build_AIGC_3K(opt, blip_tokenizer, flag="train", transform=None):
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = opt.batch_size
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = opt.batch_size

    dataset = AIGC_3K(
        opt=opt,
        blip_tokenizer=blip_tokenizer,
        flag=flag,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_flag)
    return dataset, dataloader
# endregion


class AIGCIQA_2023(Dataset):
    def __init__(self, opt, blip_tokenizer, flag="train"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.opt = opt
        if self.opt.aug:
            self.n_args = self.opt.n_args
        else:
            self.n_args = 1

        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform = transform
        self.aug = opt.aug
        self.blip_tokenizer = blip_tokenizer
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.__read_data__()

    def __read_data__(self):
        assert os.path.exists(self.opt.aigciqa_root_path), f"aigciqa_root_path not exists"
        self.aigciqa_root_path = self.opt.aigciqa_root_path
        self.img_root_folder = os.path.join(self.aigciqa_root_path, "Image")
        self.dataset_data_path = os.path.join(self.aigciqa_root_path, "DATA")
        self.xlsx_path = os.path.join(self.aigciqa_root_path, "prompts.xlsx")

        self.xlsx = pd.read_excel(self.xlsx_path)
        self.prompts = self.xlsx.iloc[:, 2]

        self.mos_quality_data = scipy.io.loadmat(os.path.join(self.dataset_data_path, "MOS", "mosz1.mat"))['MOSz']
        self.mos_authenticity_data = scipy.io.loadmat(os.path.join(self.dataset_data_path, "MOS", "mosz2.mat"))['MOSz']
        self.mos_correspondence_data = scipy.io.loadmat(os.path.join(self.dataset_data_path, "MOS", "mosz3.mat"))['MOSz']

        self.std_quality_data = scipy.io.loadmat(os.path.join(self.dataset_data_path, "STD", "SD1.mat"))['SD']
        self.std_authenticity_data = scipy.io.loadmat(os.path.join(self.dataset_data_path, "STD", "SD2.mat"))['SD']
        self.std_correspondence_data = scipy.io.loadmat(os.path.join(self.dataset_data_path, "STD", "SD3.mat"))['SD']

        self.all_img_folder = os.path.join(self.img_root_folder, "allimg")

        self.allimg_paths = glob.glob(os.path.join(self.all_img_folder, "*.png"))
        self.total_num = len(self.allimg_paths)

    def __len__(self):
        return self.total_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.allimg_paths[idx]
        assert os.path.exists(img_name), f"img_name not exists:{img_name}"
        image = Image.open(img_name)
        prompt_idx = idx % 400 // 4
        prompt = self.prompts[prompt_idx]
        text_input = self.blip_tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt")

        sample = dict()

        if self.aug:
            n_args = self.opt.n_args
            aug_images = self._augment_images(image, n_args)
            img_list = aug_images
        else:
            n_args = 1
            img_list = [image]

        aug_img_tensors = []
        if self.transform:
            for i in range(len(img_list)):
                img_list[i] = self.transform(img_list[i])
                aug_img_tensors.append(img_list[i])

        aug_img_tensors = torch.stack(aug_img_tensors, dim=0)

        sample['image'] = aug_img_tensors  # D X C X H X W

        sample['mos_quality_data'] = np.tile(self.mos_quality_data[idx], self.n_args)  # D,
        sample['mos_authenticity_data'] = np.tile(self.mos_authenticity_data[idx], self.n_args)  # D,
        sample['mos_correspondence_data'] = np.tile(self.mos_correspondence_data[idx], self.n_args)  # D,
        sample['std_quality_data'] = np.tile(self.std_quality_data[idx], self.n_args)  # D,
        sample['std_authenticity_data'] = np.tile(self.std_authenticity_data[idx], self.n_args)  # D,
        sample['std_correspondence_data'] = np.tile(self.std_correspondence_data[idx], self.n_args)  # D,

        sample['prompt_input_ids'] = text_input.input_ids.expand(self.n_args, -1)  # D X _specify_length_
        sample['prompt_attention_mask'] = text_input.attention_mask.expand(self.n_args, -1)  # D X _specify_length_

        return sample

    def _augment_images(self, img, n_args=40):
        augmentation_lists = []
        for level in range(0, 4):
            # Define the transforms
            augmentation_level_list = [
                lambda x: imjitter(x, level),
                lambda x: imblurgauss(x, level),
                lambda x: imblurlens(x, level),
                lambda x: imblurmotion(x, level),
                lambda x: imcolordiffuse(x, level),
                lambda x: imcolorshift(x, level),
                lambda x: imcolorsaturate(x, level),
                lambda x: imsaturate(x, level),
                lambda x: imcompressjpeg(x, level),
                lambda x: imnoisegauss(x, level),
                lambda x: imnoisecolormap(x, level),
                lambda x: imnoiseimpulse(x, level),
                lambda x: imnoisemultiplicative(x, level),
                lambda x: imdenoise(x, level),
                lambda x: imbrighten(x, level),
                lambda x: imdarken(x, level),
                lambda x: immeanshift(x, level),
                lambda x: imresizedist(x, level),
                lambda x: imresizedist_bilinear(x, level),
                lambda x: imresizedist_nearest(x, level),
                lambda x: imresizedist_lanczos(x, level),
                lambda x: imsharpenHi(x, level),
                lambda x: imcontrastc(x, level),
                lambda x: imcolorblock(x, level),
                lambda x: impixelate(x, level),
                lambda x: imnoneccentricity(x, level),
                # lambda x: imwarpmap(x, np.random.randn(x.size[0], x.size[1], 2)),
                lambda x: imjitter(x, level),
            ]
            augmentation_lists += augmentation_level_list

        # Randomly select n_args augmentation methods from the list
        selected_augmentations = random.sample(augmentation_lists, n_args)

        # Apply the selected augmentations to the input image
        augmented_images = []
        for augmentation in selected_augmentations:
            augmented_image = augmentation(img)
            augmented_images.append(augmented_image)

        return augmented_images


def build_AIGCIQA(opt, blip_tokenizer, flag="train", transform=None):
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = opt.batch_size
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = opt.batch_size

    dataset = AIGCIQA_2023(
        opt=opt,
        blip_tokenizer=blip_tokenizer,
        flag=flag,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_flag)
    return dataset, dataloader


if __name__ == '__main__':
    args = TrainOptions().parse()
    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    content_aware_net = ContentAwareFeats(ckpt_path=args.content_aware_ckpt_path, device=device)
    quality_aware_net = QualityAwareFeats(ckpt_path=args.quality_aware_ckpt_path, device=device)

    # text_alignment_net: last_mlp_layer: 768 -> 1
    text_alignment_net = ImageReward(ckpt_path=args.text_alignment_ckpt_path, med_config=args.text_alignment_med_config_path, device=device)
    text_alignment_tokenizer = text_alignment_net.blip.tokenizer

    test_data, test_loader = build_AIGCIQA(args,
                                           text_alignment_tokenizer,
                                           flag='test')

    with torch.no_grad():
        for i, (batch_meta) in enumerate(test_loader):
            batch_img = batch_meta["image"].float().to(device)  # bsz X D X C X H X W
            print(batch_img.shape)
            batch_mos_quality_data = batch_meta["mos_quality_data"].float().to(device)  # bsz X _type_
            print(batch_mos_quality_data.shape)
            batch_mos_authenticity_data = batch_meta["mos_authenticity_data"].float().to(device)
            print(batch_mos_authenticity_data.shape)
            batch_mos_correspondence_data = batch_meta["mos_correspondence_data"].float().to(device)
            print(batch_mos_correspondence_data.shape)
            batch_std_quality_data = batch_meta["std_quality_data"].float().to(device)
            print(batch_std_quality_data.shape)
            batch_std_authenticity_data = batch_meta["std_authenticity_data"].float().to(device)
            print(batch_std_authenticity_data.shape)
            batch_std_correspondence_data = batch_meta["std_correspondence_data"].float().to(device)
            print(batch_std_correspondence_data.shape)
