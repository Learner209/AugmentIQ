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
import os
# from loguru import logger


class ReIQAFeats(nn.Module):
    def __init__(self, args, device):
        super(ReIQAFeats, self).__init__()
        self.args = TrainOptions().parse()
        self.device = device
        self.model = None
        self.args = args
        assert hasattr(self.args, "content_aware_ckpt_path") and os.path.exists(self.args.content_aware_ckpt_path), "content_aware_ckpt_path not found"
        assert hasattr(self.args, "quality_aware_ckpt_path") and os.path.exists(self.args.content_aware_ckpt_path), "quality_aware_ckpt_path not found"

        self.content_aware_net = self.load_model(self.args.content_aware_ckpt_path)
        self.quality_aware_net = self.load_model(self.args.quality_aware_ckpt_path)
        self.train_mode()

        self.content_aware_feat_head = nn.Sequential(
            nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256)
        ).to(self.device)
        # initial MLP param
        for name, param in self.content_aware_feat_head.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (256 + 1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)

        self.quality_aware_feat_head = nn.Sequential(
            nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256)
        ).to(self.device)
        # initial MLP param
        for name, param in self.quality_aware_feat_head.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (256 + 1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)

        self.final_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(self.device)

        # region
        self.content_aware_net_head = nn.Sequential(
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
        for name, param in self.content_aware_net_head.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (256 + 1))
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
        # endregion

    def load_model(self, ckpt_path):
        # build model
        model = RGBSingleHead("resnet50", "mlp", 128, None)

        # check and resume a model
        assert os.path.exists(ckpt_path), "checkpoint path not exists"
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['model'])

        model.to(self.device)
        model.train()

        return model

    def forward(self, img_tensor):
        bsz, C, H, W = img_tensor.shape

        half_img_tensor = F.interpolate(img_tensor, size=(H // 2, W // 2), mode='bilinear', align_corners=False)  # half-scale

        content_aware_feat_full = self.content_aware_net.module.encoder(img_tensor.to(self.device))
        content_aware_feat_full = self.content_aware_net.module.head(content_aware_feat_full)

        content_aware_feat_half = self.content_aware_net.module.encoder(half_img_tensor.to(self.device))
        content_aware_feat_half = self.content_aware_net.module.head(content_aware_feat_half)

        content_aware_feat = torch.cat((content_aware_feat_full, content_aware_feat_half), dim=1)  # *  bsz * 256
        # content_aware_feat = F.normalize(content_aware_feat, dim=1)
        content_aware_feat = self.content_aware_feat_head(content_aware_feat)

        quality_aware_feat_full = self.quality_aware_net.module.encoder(img_tensor.to(self.device))
        quality_aware_feat_full = self.quality_aware_net.module.head(quality_aware_feat_full)

        quality_aware_feat_half = self.quality_aware_net.module.encoder(half_img_tensor.to(self.device))
        quality_aware_feat_half = self.quality_aware_net.module.head(quality_aware_feat_half)

        quality_aware_feat = torch.cat((quality_aware_feat_full, quality_aware_feat_half), dim=1)  # *  bsz * 256
        # quality_aware_feat = F.normalize(quality_aware_feat, dim=1)
        quality_aware_feat = self.quality_aware_feat_head(quality_aware_feat)

        final_reward = self.final_head(content_aware_feat + quality_aware_feat)
        return final_reward

    def train_mode(self):
        for param in self.content_aware_net.module.encoder.parameters():
            param.requires_grad = False
        for param in self.content_aware_net.module.head.parameters():
            param.requires_grad = True
        for param in self.quality_aware_net.module.encoder.parameters():
            param.requires_grad = False
        for param in self.quality_aware_net.module.head.parameters():
            param.requires_grad = True

    def eval_mode(self):
        for param in self.content_aware_net.module.encoder.parameters():
            param.requires_grad = False
        for param in self.content_aware_net.module.head.parameters():
            param.requires_grad = False
        for param in self.quality_aware_net.module.encoder.parameters():
            param.requires_grad = False
        for param in self.quality_aware_net.module.head.parameters():
            param.requires_grad = False
