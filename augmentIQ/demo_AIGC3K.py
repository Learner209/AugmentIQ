import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from torchvision import transforms
from torchvision.models import resnet50
import torch.optim as optim
import torch.nn as nn

import argparse
from collections import defaultdict
from augmentIQ.datasets.iqa_dataset import build_AIGC_3K, AIGC_3K
import copy
import os
import random

import time
from augmentIQ.utils.metric import metric

from augmentIQ.utils.tools import EarlyStopping, adjust_learning_rate
from loguru import logger

import torch.utils.data.distributed
import torch.multiprocessing as mp

from augmentIQ.options.train_options import TrainOptions
from typing import List, Tuple
from einops import rearrange, repeat

from augmentIQ.content_aware_feats import ContentAwareFeats
from augmentIQ.quality_aware_feats import QualityAwareFeats
from ImageReward.ImageReward import ImageReward
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import json
from scipy.stats import spearmanr, kendalltau, pearsonr

from torchmetrics import SpearmanCorrCoef
from torchmetrics.regression import PearsonCorrCoef


import warnings
warnings.filterwarnings("ignore")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


class IQA_trainer(object):
    def __init__(self, args, setting):
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.content_aware_net, self.quality_aware_net, self.text_alignment_net = None, None, None
        self._build_model()
        self.model_cards = [self.content_aware_net, self.quality_aware_net, self.text_alignment_net]
        assert self.model_cards[0] is not None and self.model_cards[1] is not None and self.model_cards[2] is not None

        self.writer = SummaryWriter(log_dir=os.path.join("runs", setting))

    def _build_model(self):
        # content_aware_net, quality_aware_net: feat_dim = 128 X 2
        self.content_aware_net = ContentAwareFeats(ckpt_path=self.
                                                   args.content_aware_ckpt_path, device=self.device)
        self.quality_aware_net = QualityAwareFeats(ckpt_path=self.args.quality_aware_ckpt_path, device=self.device)

        # text_alignment_net: last_mlp_layer: 768 -> 1
        self.text_alignment_net = ImageReward(ckpt_path=self.args.text_alignment_ckpt_path, med_config=self.args.text_alignment_med_config_path, device=self.device)
        self.text_alignment_tokenizer = self.text_alignment_net.blip.tokenizer

        content_aware_net_trainable_params = sum(p.numel() for p in self.content_aware_net.parameters() if p.requires_grad)
        logger.info('Content aware net trainable parameters: {}'.format(content_aware_net_trainable_params))
        quality_aware_net_trainable_params = sum(p.numel() for p in self.quality_aware_net.parameters() if p.requires_grad)
        logger.info('Quality aware net trainable parameters: {}'.format(quality_aware_net_trainable_params))
        text_alignment_net_trainable_parameters = sum(p.numel() for p in self.text_alignment_net.parameters() if p.requires_grad)
        logger.info('Text alignment trainable parameters: {}'.format(text_alignment_net_trainable_parameters))

    def _select_optimizer(self):
        re_iqa_model_optim = optim.Adam(list(self.content_aware_net.parameters()) + list(self.quality_aware_net.parameters()), lr=self.args.learning_rate)
        text_alignment_model_optim = optim.Adam(self.text_alignment_net.parameters(), lr=self.args.learning_rate)
        return re_iqa_model_optim, text_alignment_model_optim

    def train_mode(self):
        self.content_aware_net.train_mode()
        self.quality_aware_net.train_mode()
        self.text_alignment_net.train_mode()

    def eval_mode(self):
        self.content_aware_net.eval_mode()
        self.quality_aware_net.eval_mode()
        self.text_alignment_net.eval_mode()

    def _select_criterion(self):
        re_iqa_criterion = nn.MSELoss()
        # re_iqa_criterion = nn.CrossEntropyLoss()
        text_alignment_criterion = nn.MSELoss()
        # re_iqa_criterion = nn.CrossEntropyLoss()
        return re_iqa_criterion, text_alignment_criterion

    def vali(self, vali_data, vali_loader, fold):
        self.eval_mode()

        re_iqa_valid_loss = []
        text_alignment_valid_loss = []

        re_iqa_criterion, text_alignment_criterion = self._select_criterion()

        with torch.no_grad():
            for i, (batch_meta) in enumerate(vali_loader):

                re_iqa_pred, re_iqa_true, text_alignment_pred, text_alignment_true = self._process_one_batch(batch_meta)

                spcc_metric = SpearmanCorrCoef().to(self.device)
                plcc_metric = PearsonCorrCoef().to(self.device)

                # re_iqa_loss = - spcc_metric(re_iqa_pred, re_iqa_true)  # - 2 * plcc_metric(re_iqa_pred, re_iqa_true)
                # text_alignment_loss = - spcc_metric(text_alignment_pred, text_alignment_true)  # - 2 * plcc_metric(text_alignment_pred, text_alignment_true)

                re_iqa_loss = re_iqa_criterion(re_iqa_pred, re_iqa_true)
                text_alignment_loss = text_alignment_criterion(text_alignment_pred, text_alignment_true)

                # logger.info("ReIQA valid loss:{0:.7f} | Text alignment valid loss:{1:.7f}.".format(re_iqa_loss.item(), text_alignment_loss.item()))
                re_iqa_valid_loss.append(re_iqa_loss.detach().item())
                text_alignment_valid_loss.append(text_alignment_loss.detach().item())
                self.writer.add_scalar(f'{fold}/re_iqa/valid_loss', re_iqa_loss.item(), global_step=i)
                self.writer.add_scalar(f'{fold}/text_alignment/valid_loss', text_alignment_loss.item(), global_step=i)

                if (i + 1) % self.args.save_freq == 0:
                    batch_img = batch_meta["image"].float().to(self.device)  # bsz X D X C X H X W
                    batch_img = torch.flatten(batch_img, end_dim=1)
                    assert batch_img.dim() == 4
                    # self.writer.add_images('img/vali', batch_img, global_step=(i + 1) // self.args.save_freq)

        re_iqa_valid_loss = np.average(re_iqa_valid_loss)
        text_alignment_valid_loss = np.average(text_alignment_valid_loss)

        return re_iqa_valid_loss, text_alignment_valid_loss

    def train(self, setting):
        train_data, train_loader = build_AIGC_3K(self.args, self.text_alignment_tokenizer, flag='train')

        self.train_mode()

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(loss_cnt=2, model_cnt=3, patience=self.args.tolerance, verbose=True)

        re_iqa_optim, text_alignment_optim = self._select_optimizer()
        re_iqa_criterion, text_alignment_criterion = self._select_criterion()

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        best_model_cards = None
        best_vali_loss = float('inf')
        # best_test_loss = float('inf')

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
            logger.info(f"Fold {fold+1}")
            train_fold = torch.utils.data.Subset(train_data, train_idx)
            val_fold = torch.utils.data.Subset(train_data, val_idx)

            train_loader_fold = DataLoader(
                train_fold,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                drop_last=False)

            val_loader_fold = DataLoader(
                val_fold,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                drop_last=False)

            for epoch in range(self.args.epochs):
                time_now = time.time()
                iter_count = 0

                re_iqa_train_loss = []
                text_alignment_train_loss = []

                self.train_mode()

                epoch_time = time.time()
                for i, (batch_meta) in enumerate(train_loader_fold):
                    iter_count += 1

                    re_iqa_optim.zero_grad()
                    text_alignment_optim.zero_grad()

                    re_iqa_pred, re_iqa_true, text_alignment_pred, text_alignment_true = self._process_one_batch(batch_meta)

                    spcc_metric = SpearmanCorrCoef().to(self.device)
                    plcc_metric = PearsonCorrCoef().to(self.device)

                    # re_iqa_loss = -spcc_metric(re_iqa_pred, re_iqa_true)  # - 2 * plcc_metric(re_iqa_pred, re_iqa_true)
                    # text_alignment_loss = -spcc_metric(text_alignment_pred, text_alignment_true)  # - 2 * plcc_metric(text_alignment_pred, text_alignment_true)

                    re_iqa_loss = re_iqa_criterion(re_iqa_pred, re_iqa_true)
                    text_alignment_loss = text_alignment_criterion(text_alignment_pred, text_alignment_true)

                    logger.info("ReIQA training loss:{:.7f} | Text alignment training loss:{:.7f}.".format(re_iqa_loss.item(), text_alignment_loss.item()))
                    re_iqa_train_loss.append(re_iqa_loss.detach().item())
                    text_alignment_train_loss.append(text_alignment_loss.detach().item())

                    # Log the training loss to TensorBoard
                    global_step = epoch * train_steps + i
                    self.writer.add_scalar(f'{fold}/re_iqa/train_loss', re_iqa_loss.item(), global_step=global_step)
                    self.writer.add_scalar(f'{fold}/text_alignment/train_loss', text_alignment_loss.item(), global_step=global_step)

                    if (i + 1) % 100 == 0:
                        logger.info("\titers: {0}, epoch: {1} | re_iqa_loss: {2:.7f} | text_alignment_loss: {3:.7f}".format(i + 1, epoch + 1, re_iqa_loss.item(), text_alignment_loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.epochs - epoch) * train_steps - i)
                        logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                        batch_img = batch_meta["image"].float().to(self.device)  # bsz X D X C X H X W
                        batch_img = torch.flatten(batch_img, end_dim=1)
                        assert batch_img.dim() == 4
                        # self.writer.add_images('img/train', batch_img, global_step=epoch * (train_steps + 1) // 100 + (i + 1) // 100)

                    re_iqa_loss.backward()
                    text_alignment_loss.backward()

                    re_iqa_optim.step()
                    text_alignment_optim.step()

                    self.test(setting, fold)

                    logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

                re_iqa_train_loss = np.average(re_iqa_train_loss)
                text_alignment_train_loss = np.average(text_alignment_train_loss)

                re_iqa_valid_loss, text_alignment_valid_loss = self.vali(val_fold, val_loader_fold, fold)

                logger.info("Epoch: {0}, Steps: {1} | Re_IQA train Loss: {2:.7f} | Text alignment train loss: {3:.7f} | \
                            ReIQA vali Loss: {4:.7f} | Text alignment vali Loss: {5:.7f}".format(
                    epoch + 1, train_steps, re_iqa_train_loss, text_alignment_train_loss, re_iqa_valid_loss, text_alignment_valid_loss))

                early_stopping([re_iqa_valid_loss, text_alignment_valid_loss], self.model_cards, path)
                adjust_learning_rate([re_iqa_optim, text_alignment_optim], epoch + 1, self.args)
                if early_stopping.early_stop:
                    logger.info("Early stopping")
                    break

            if early_stopping.best_score < best_vali_loss:
                best_vali_loss = early_stopping.val_loss_min
                best_model_cards = [copy.deepcopy(model) for model in self.model_cards]

            # Test the model
            self.test(setting, fold)

            best_model_paths = [path + '/' + f'checkpoint_{i}.pth' for i in range(len(self.model_cards))]
            [best_model.load_state_dict(torch.load(best_model_path)) for (best_model_path, best_model) in zip(best_model_paths, best_model_cards)]
            state_dicts = [best_model.state_dict() for best_model in best_model_cards]
            [torch.save(state_dict, path + '/' + f'checkpoint_{i}.pth') for (i, state_dict) in enumerate(state_dicts)]

            return best_model_cards

    def _process_one_batch(self, batch_meta):
        batch_img = batch_meta["image"].float().to(self.device)  # bsz X D X C X H X W
        assert batch_img.dim() == 5
        batch_img = rearrange(batch_img, 'bsz D C H W -> (bsz D) C H W')
        assert batch_img.dim() == 4

        batch_mos_quality = batch_meta["mos_quality"].float().to(self.device)  # bsz X _type_
        batch_mos_quality = F.normalize(batch_mos_quality, dim=0).reshape(-1, 1)
        batch_std_quality = batch_meta["std_quality"].float().to(self.device)
        batch_std_quality = F.normalize(batch_std_quality, dim=0).reshape(-1, 1)
        batch_mos_align = batch_meta["mos_align"].float().to(self.device)
        batch_mos_align = F.normalize(batch_mos_align, dim=0).reshape(-1, 1)
        batch_std_align = batch_meta["std_align"].float().to(self.device)
        batch_std_align = F.normalize(batch_std_align, dim=0).reshape(-1, 1)

        batch_prompt_input_ids = batch_meta["prompt_input_ids"].long().to(self.device)
        batch_prompt_input_ids = batch_prompt_input_ids.reshape(-1, batch_prompt_input_ids.shape[-1])
        batch_prompt_attention_mask = batch_meta["prompt_attention_mask"].long().to(self.device)
        batch_prompt_attention_mask = batch_prompt_attention_mask.reshape(-1, batch_prompt_attention_mask.shape[-1])

        # logger.info(batch_img.shape, batch_mos_quality.shape, batch_std_quality.shape, batch_mos_align.shape, batch_std_align.shape)
        content_aware_feat = self.content_aware_net(batch_img)
        assert content_aware_feat.dim() == 2
        # content_aware_feat = rearrange(content_aware_feat, '(bsz D) d_model -> bsz D d_model', bsz=batch_img.shape[0])
        quality_aware_feat = self.quality_aware_net(batch_img)
        assert quality_aware_feat.dim() == 2
        # quality_aware_feat = rearrange(quality_aware_feat, '(bsz D) d_model -> bsz D d_model', bsz=batch_img.shape[0])
        feat = content_aware_feat + quality_aware_feat

        text_alignment_feat = self.text_alignment_net(batch_img, batch_prompt_input_ids, batch_prompt_attention_mask)
        assert text_alignment_feat.dim() == 2

        # logger.info(f"feat.shape:{feat.shape}, batch_std_quality.shape:{batch_std_quality.shape}, text_alignment_feat.shape:{text_alignment_feat.shape}, batch_std_align.shape:{batch_std_align.shape}")
        # return feat, batch_std_quality, text_alignment_feat, batch_std_align
        return feat, batch_mos_quality, text_alignment_feat, batch_mos_align

    def test(self, setting, fold=0):
        test_data, test_loader = build_AIGC_3K(self.args, self.text_alignment_tokenizer, flag='test')

        self.eval_mode()

        re_iqa_metrics_all = []
        text_alignment_metrics_all = []
        instance_num = 0

        with torch.no_grad():
            for i, (batch_meta) in enumerate(test_loader):
                re_iqa_pred, re_iqa_true, text_alignment_pred, text_alignment_true = self._process_one_batch(batch_meta)

                batch_size = re_iqa_pred.shape[0]

                instance_num += batch_size

                re_iqa_pred = re_iqa_pred.reshape(-1)
                re_iqa_true = re_iqa_true.reshape(-1)
                text_alignment_pred = text_alignment_pred.reshape(-1)
                text_alignment_true = text_alignment_true.reshape(-1)

                re_iqa_batch_metric = np.array(metric(re_iqa_pred, re_iqa_true)) * batch_size
                text_alignment_batch_metric = np.array(metric(text_alignment_pred, text_alignment_true)) * batch_size

                re_iqa_metrics_all.append(re_iqa_batch_metric)
                text_alignment_metrics_all.append(text_alignment_batch_metric)

                if (i + 1) % self.args.save_freq == 0:
                    batch_img = batch_meta["image"].float().to(self.device)  # bsz X D X C X H X W
                    batch_img = torch.flatten(batch_img, end_dim=1)
                    # self.writer.add_images('img/test', batch_img, global_step=(i + 1) // self.args.save_freq)

        logger.info('test instance num: {}'.format(instance_num))

        re_iqa_metrics_all = np.stack(re_iqa_metrics_all, axis=0)
        text_alignment_metrics_all = np.stack(text_alignment_metrics_all, axis=0)
        re_iqa_metrics_mean = re_iqa_metrics_all.sum(axis=0) / instance_num
        text_alignment_metrics_mean = text_alignment_metrics_all.sum(axis=0) / instance_num

        # result save
        folder_path = './forecast_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        spcc, plcc = re_iqa_metrics_mean
        logger.info(f"The Re_IQA metrics ->: spcc:{spcc}, plcc:{plcc}")
        self.writer.add_scalar(f'{fold}/re_iqa/srcc_test_loss', spcc, global_step=0)
        self.writer.add_scalar(f'{fold}/re_iqa/plcc_test_loss', plcc, global_step=0)

        spcc, plcc = text_alignment_metrics_mean
        logger.info(f"The text alignment metrics ->: spcc:{spcc}, plcc:{plcc}")
        self.writer.add_scalar(f'{fold}/text_alignment/srcc_test_loss', spcc, global_step=0)
        self.writer.add_scalar(f'{fold}/text_alignment/plcc_test_loss', plcc, global_step=0)

        self.train_mode()

        return np.sum(re_iqa_metrics_mean), np.sum(text_alignment_metrics_mean)


if __name__ == "__main__":
    args = TrainOptions().parse()

    # random seed
    fix_seed = args.random_seed
    torch.manual_seed(fix_seed)
    random.seed(fix_seed)
    np.random.seed(fix_seed)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        logger.warning(
            f"Using multi-gpu support, the device ids are {args.device_ids}")

    logger.info('Args in experiment:')
    args.is_training = True
    logger.info(args)
    args.aug = True
    setting = "AIGC3K_bsz_{}_n_args_{}_epochs_{}_aug_{}".format(
        args.batch_size, args.n_args, args.epochs, args.aug
    )
    args.tb_path = "runs/"

    if not args.is_training:
        logger.info(setting)
        exp = IQA_trainer(args, setting)
        exp.test(setting)
    else:
        exp = IQA_trainer(args, setting)

        logger.info('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        logger.info(setting)
        exp.train(setting=setting)

        logger.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(args)
