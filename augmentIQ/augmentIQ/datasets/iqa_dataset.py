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
import torch.nn as nn
from torch.utils.data import DataLoader

from loguru import logger

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data.distributed
import torch.multiprocessing as mp

import sys
sys.path.append("/home/liilu/Desktop/COURSE/Pic_proc/AIGC/BASELINE/augmentIQ")

from augmentIQ.options.train_options import TrainOptions
from augmentIQ.datasets.iqa_distortions import *

from augmentIQ.reiqa_feats import ReIQAFeats
from ImageReward.ImageReward import ImageReward
import scipy.io

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import glob
import sys
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _convert_image_to_bgr(image):
    return image.convert("RGB")


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

        mos_quality = self.data_frame.at[idx, "mos_quality"]
        std_quality = self.data_frame.at[idx, "std_quality"]
        mos_align = self.data_frame.at[idx, "mos_align"]
        std_align = self.data_frame.at[idx, "std_align"]
        assert isinstance(mos_quality, float) and isinstance(std_quality, float) and isinstance(mos_align, float) and isinstance(std_align, float)

        sample = dict()

        transform = transforms.Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        sample['image'] = transform(image)  # D X C X H X W

        sample['mos_quality'] = mos_quality  # 1
        sample['std_quality'] = std_quality  # 1
        sample['mos_align'] = mos_align  # 1
        sample['std_align'] = std_align  # 1
        sample['prompt_input_ids'] = text_input.input_ids.squeeze(0)  # 1 X _specify_length_
        sample['prompt_attention_mask'] = text_input.attention_mask.squeeze(0)  # 1 X _specify_length_
        return sample


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
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
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
        del self.allimg_paths
        assert not hasattr(self, "allimg_paths")
        assert self.total_num == 2400, f"total_num not 2400, but {self.total_num}"

    def __len__(self):
        return self.total_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.all_img_folder, f"{idx}.png")
        assert os.path.exists(img_name), f"img_name not exists:{img_name}"
        image = Image.open(img_name)
        prompt_idx = (idx % 400) // 4
        prompt = self.prompts[prompt_idx]
        text_input = self.blip_tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt")
        # logger.warning(f"The image path is {img_name} and the idx is {idx} and the prompt idx is {prompt_idx} and the prompt is {prompt}")
        sample = dict()

        transform = transforms.Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        sample['image'] = transform(image)  # C X H X W

        sample['mos_quality_data'] = self.mos_quality_data[idx]  # 1
        sample['mos_authenticity_data'] = self.mos_authenticity_data[idx]  # 1
        sample['mos_correspondence_data'] = self.mos_correspondence_data[idx]  # 1
        sample['std_quality_data'] = self.std_quality_data[idx]  # 1
        sample['std_authenticity_data'] = self.std_authenticity_data[idx]  # 1
        sample['std_correspondence_data'] = self.std_correspondence_data[idx]  # 1

        sample['prompt_input_ids'] = text_input.input_ids.squeeze(0)  # 1 X _specify_length_
        sample['prompt_attention_mask'] = text_input.attention_mask.squeeze(0)  # 1 X _specify_length_

        return sample


def build_dataset(opt, blip_tokenizer, flag="train", transform=None, dataset_card="AIGCIQA_2023"):
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = opt.batch_size
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = opt.batch_size

    if dataset_card == "AIGCIQA2023":
        dataset = AIGCIQA_2023(
            opt=opt,
            blip_tokenizer=blip_tokenizer,
            flag=flag,
        )
    else:
        dataset = AIGC_3K(
            opt=opt,
            blip_tokenizer=blip_tokenizer,
            flag=flag,
        )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_flag)
    return dataset, dataloader


if __name__ == '__main__':
    args = TrainOptions().parse()
    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    content_aware_net = ReIQAFeats(args=args, device=device)

    text_alignment_net = ImageReward(ckpt_path=args.text_alignment_ckpt_path, med_config=args.text_alignment_med_config_path, device=device)
    text_alignment_tokenizer = text_alignment_net.blip.tokenizer

    test_data, test_loader = build_dataset(args, text_alignment_tokenizer, flag='test', dataset_card="AIGCIQA2023")

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
