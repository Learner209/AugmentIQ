from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data.distributed
import torch.multiprocessing as mp

from augmentIQ.options.train_options import TrainOptions
from torch.utils.data import DataLoader
from torch.utils import data
from PIL import Image
from torchvision import transforms
import csv
import os
import scipy.io
import numpy as np
import time
import subprocess
import pandas as pd
import pickle
from augmentIQ.networks.build_backbone import RGBSingleHead
import torch.nn.functional as F
from einops import rearrange, repeat


class QualityAwareFeats(nn.Module):
    def __init__(self, ckpt_path, device):
        super(QualityAwareFeats, self).__init__()
        self.args = TrainOptions().parse()
        self.device = device
        self.model = None
        self.load_model(ckpt_path)

        self.feat_head = nn.Sequential(
            nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128)
        ).to(self.device)
        # initial MLP param
        for name, param in self.feat_head.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (128 + 1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)

        self.quality_aware_net_head = nn.Sequential(
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.LayerNorm(128),  # Add here
            nn.ReLU(),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.LayerNorm(64),  # Add here
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        # initial MLP param
        for name, param in self.quality_aware_net_head.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (256 + 1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)

    def load_model(self, ckpt_path):
        # build model
        self.model = RGBSingleHead("resnet50", "mlp", 128, None)

        # check and resume a model
        assert os.path.exists(ckpt_path), "checkpoint path not exists"
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(checkpoint['model'])

        self.model.to(self.device)
        self.model.train()

        for param in self.model.module.encoder.parameters():
            param.requires_grad = False
        for param in self.model.module.head.parameters():
            param.requires_grad = True

    def forward(self, img_tensor):
        bsz, C, H, W = img_tensor.shape

        img_tensor_2 = F.interpolate(img_tensor, size=(H // 2, W // 2), mode='bilinear', align_corners=False)  # half-scale
        feat1 = self.model.module.encoder(img_tensor.to(self.device))
        feat1 = self.model.module.head(feat1)
        feat1 = self.feat_head(feat1)  # + feat1
        feat2 = self.model.module.encoder(img_tensor_2.to(self.device))
        feat2 = self.model.module.head(feat2)
        feat2 = self.feat_head(feat2)  # + feat2
        feat = torch.cat((feat1, feat2), dim=1)
        feat = F.normalize(feat, dim=1)
        feat = self.quality_aware_net_head(feat)
        return feat

    def train_mode(self):
        for param in self.model.module.encoder.parameters():
            param.requires_grad = False
        for param in self.model.module.head.parameters():
            param.requires_grad = True

    def eval_mode(self):
        for param in self.model.module.encoder.parameters():
            param.requires_grad = False
        for param in self.model.module.head.parameters():
            param.requires_grad = False
