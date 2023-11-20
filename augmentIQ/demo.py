import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import torch.optim as optim
import torch.nn as nn

from augmentIQ.datasets.iqa_dataset import build_dataset
import copy
import os
import random

import time
from augmentIQ.utils.metric import metric

from augmentIQ.utils.tools import EarlyStopping, adjust_learning_rate
from loguru import logger

import torch.utils.data.distributed
import torch.nn.functional as F

from augmentIQ.reiqa_feats import ReIQAFeats
from augmentIQ.options.train_options import TrainOptions

from ImageReward.ImageReward import ImageReward
from torch.utils.tensorboard import SummaryWriter
import json
from torchmetrics import SpearmanCorrCoef
from torchmetrics.regression import PearsonCorrCoef
from scipy.stats import spearmanr, pearsonr

import sys
import warnings
warnings.filterwarnings("ignore")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class IQA_trainer(object):
    def __init__(self, args, setting):
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
        self._build_model()

        self.writer = SummaryWriter(log_dir=os.path.join("runs", setting))
        assert hasattr(self.args, "dataset_card")
        assert self.args.dataset_card in ["AIGC3K", "AIGCIQA2023"]
        assert hasattr(self.args, "model_card")
        assert self.args.model_card in ["text_alignment", "content_quality_aware"]

    def _build_model(self):

        # content_aware_net, quality_aware_net: feat_dim = 128 X 2
        self.reiqa_net = ReIQAFeats(self.args, device=self.device)

        self.text_alignment_net = ImageReward(ckpt_path=self.args.text_alignment_ckpt_path, med_config=self.args.text_alignment_med_config_path, device=self.device)
        self.text_alignment_tokenizer = self.text_alignment_net.blip.tokenizer

        text_alignment_net_trainable_parameters = sum(p.numel() for p in self.text_alignment_net.parameters() if p.requires_grad)
        logger.info('Text alignment trainable parameters: {}'.format(text_alignment_net_trainable_parameters))

    def _select_optimizer(self):
        return optim.Adam(self.text_alignment_net.parameters(), lr=self.args.learning_rate)

    def train_mode(self):
        self.text_alignment_net.train_mode()

    def eval_mode(self):
        self.text_alignment_net.eval_mode()

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, fold):
        self.eval_mode()

        vali_loss = []
        criterion = self._select_criterion()

        with torch.no_grad():
            for i, (batch_meta) in enumerate(vali_loader):

                pred, true = self._process_one_batch(batch_meta)

                spcc_metric = SpearmanCorrCoef().to(self.device)
                plcc_metric = PearsonCorrCoef().to(self.device)
                spcc_loss = spcc_metric(pred, true)
                plcc_loss = plcc_metric(pred, true)
                logger.info(f"With the torchmetrics standard, the valid loss is {spcc_loss}, {plcc_loss}")
                # re_iqa_loss = - spcc_metric(re_iqa_pred, re_iqa_true)  # - 2 * plcc_metric(re_iqa_pred, re_iqa_true)
                # text_alignment_loss = - spcc_metric(text_alignment_pred, text_alignment_true)  # - 2 * plcc_metric(text_alignment_pred, text_alignment_true)

                loss = criterion(pred, true)
                # logger.info("ReIQA valid loss:{0:.7f} | Text alignment valid loss:{1:.7f}.".format(re_iqa_loss.item(), text_alignment_loss.item()))

                vali_loss.append(loss.detach().item())
                self.writer.add_scalar(f'{fold}/text_alignment/valid_loss', loss.item(), global_step=i)
        vali_loss = np.average(vali_loss)

        return vali_loss

    def train(self, setting):
        train_data, train_loader = build_dataset(self.args, self.text_alignment_tokenizer, flag='train', dataset_card=self.args.dataset_card)

        self.train_mode()

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=3, verbose=True)

        optim = self._select_optimizer()
        criterion = self._select_criterion()

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        best_vali_loss = float('inf')

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

                train_loss = []

                self.train_mode()

                epoch_time = time.time()
                for i, (batch_meta) in enumerate(train_loader_fold):
                    iter_count += 1

                    optim.zero_grad()

                    pred, true = self._process_one_batch(batch_meta)

                    spcc_metric = SpearmanCorrCoef().to(self.device)
                    plcc_metric = PearsonCorrCoef().to(self.device)
                    spcc_loss = spcc_metric(pred, true)
                    plcc_loss = plcc_metric(pred, true)

                    if self.args.model_card == "text_alignment":
                        loss = - spcc_loss - plcc_loss
                    else:
                        loss = criterion(pred, true)

                    train_loss.append(loss.detach().item())

                    global_step = epoch * train_steps + i
                    self.writer.add_scalar(f'{fold}/text_alignment/train_loss', loss.item(), global_step=global_step)

                    if (i + 1) % 100 == 0:
                        logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.epochs - epoch) * train_steps - i)
                        logger.info('\tspeed: {:.0f}s/iter; left time: {:.1f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    loss.backward()
                    optim.step()

                train_loss = np.average(train_loss)
                vali_loss = self.vali(val_fold, val_loader_fold, fold)

                # self.test(setting, fold)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))

                adjust_learning_rate(optim, epoch + 1, self.args)

                early_stopping(vali_loss, self.text_alignment_net, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            if early_stopping.best_score < best_vali_loss:
                best_vali_loss = early_stopping.val_loss_min

            # Test the model
            self.test(setting, fold)
            if self.args.model_card == "text_alignment":
                state_dict = self.text_alignment_net.state_dict()
            else:
                state_dict = self.reiqa_net.state_dict()

            torch.save(state_dict, path + '/' + f"{setting}.pth")

    def _process_one_batch(self, batch_meta):

        assert self.args.dataset_card in ["AIGC3K", "AIGCIQA2023"]
        assert self.args.model_card in ["text_alignment", "content_quality_aware"]

        batch_img = batch_meta["image"].float().to(self.device)  # bsz X C X H X W
        assert batch_img.dim() == 4 and batch_img.shape[1] == 3 and batch_img.shape[2] == 224 and batch_img.shape[3] == 224
        bsz = batch_img.shape[0]

        pred, true = None, None
        if self.args.model_card == "text_alignment":
            batch_prompt_input_ids = batch_meta["prompt_input_ids"].long().to(self.device)
            assert batch_prompt_input_ids.dim() == 2 and batch_prompt_input_ids.shape[0] == bsz, f"{batch_prompt_input_ids.shape}"
            # batch_prompt_input_ids = batch_prompt_input_ids.reshape(-1, batch_prompt_input_ids.shape[-1])
            batch_prompt_attention_mask = batch_meta["prompt_attention_mask"].long().to(self.device)
            assert batch_prompt_attention_mask.dim() == 2 and batch_prompt_input_ids.shape[0] == bsz, f"{batch_prompt_attention_mask.shape}"
            # batch_prompt_attention_mask = batch_prompt_attention_mask.reshape(-1, batch_prompt_attention_mask.shape[-1])
            pred = self.text_alignment_net(batch_img, batch_prompt_input_ids, batch_prompt_attention_mask)

            if self.args.dataset_card == "AIGC3K":
                # * Achieve the ideal result, without the normallization both for pred and true
                batch_mos_align = batch_meta["mos_align"].float().unsqueeze(-1).to(self.device)  # bsz X 1
                batch_mos_align = F.normalize(batch_mos_align, dim=0).reshape(-1, 1)  # bsz X 1
                # batch_mos_align = F.normalize(batch_mos_align, dim=0)

                batch_std_align = batch_meta["std_align"].float().unsqueeze(-1).to(self.device)
                # batch_std_align = F.normalize(batch_std_align, dim=0).reshape(-1, 1)
                true = batch_mos_align
            elif self.args.dataset_card == "AIGCIQA2023":
                batch_mos_align = batch_meta["mos_correspondence_data"].float().to(self.device)
                assert batch_mos_align.dim() == 2 and batch_mos_align.shape[0] == bsz, f"{batch_mos_align.shape}"
                # batch_mos_align = F.normalize(batch_mos_align, dim=0)

                batch_std_correspondence_data = batch_meta["std_correspondence_data"].float().to(self.device)
                assert batch_std_correspondence_data.dim() == 2 and batch_std_correspondence_data.shape[0] == bsz, f"{batch_std_correspondence_data.shape}"
                batch_std_correspondence_data = F.normalize(batch_std_correspondence_data, dim=0).reshape(-1, 1)

                # print(batch_std_correspondence_data.shape)
                true = batch_mos_align
            else:
                raise NotImplementedError
            # logger.info(f"The distance between pred and true in the `process_one_batch` device is {nn.MSELoss()(pred, true)}")
            return pred, true
        else:
            # logger.info(batch_img.shape, batch_mos_quality.shape, batch_std_quality.shape, batch_mos_align.shape, batch_std_align.shape)
            feat = self.reiqa_net(batch_img)
            assert feat.dim() == 2 and feat.shape[0] == bsz and feat.shape[1] == 1, f"{feat.shape}"
            pred = feat
            if self.args.dataset_card == "AIGC3K":
                batch_mos_quality = batch_meta["mos_quality"].float().unsqueeze(-1).to(self.device)  # bsz X _type_
                assert batch_mos_quality.dim() == 2 and batch_mos_quality.shape[0] == bsz and batch_mos_quality.shape[-1] == 1, f"{batch_mos_quality.shape}"
                batch_mos_quality = F.normalize(batch_mos_quality, dim=0)

                batch_std_quality = batch_meta["std_quality"].float().unsqueeze(-1).to(self.device)
                # batch_std_quality = F.normalize(batch_std_quality, dim=0).reshape(-1, 1)

                true = batch_mos_quality
            elif self.args.dataset_card == "AIGCIQA2023":
                batch_std_quality_data = batch_meta["std_quality_data"].float().to(self.device)
                assert batch_std_quality_data.dim() == 2 and batch_std_quality_data.shape[0] == bsz, f"{batch_std_quality_data.shape}"
                # batch_std_quality_data = F.normalize(batch_std_quality_data, dim=0).reshape(-1, 1)
                # print(batch_std_quality_data.shape)
                batch_mos_quality_data = batch_meta["mos_quality_data"].float().to(self.device)  # bsz X _type_
                batch_mos_quality_data = F.normalize(batch_mos_quality_data, dim=0)
                # print(batch_mos_quality_data.shape)
                batch_mos_authenticity_data = batch_meta["mos_authenticity_data"].float().to(self.device)
                # batch_mos_authenticity_data = F.normalize(batch_mos_authenticity_data, dim=0).reshape(-1, 1)
                # print(batch_mos_authenticity_data.shape)
                true = batch_mos_quality_data
            else:
                raise NotImplementedError
            # logger.info(f"The distance between pred and true in the `process_one_batch` device is {nn.MSELoss()(pred, true)}")
            return pred, true

    def test(self, setting, fold=0):
        test_data, test_loader = build_dataset(self.args, self.text_alignment_tokenizer, flag='test', dataset_card=self.args.dataset_card)

        self.eval_mode()

        batch_metrics_all = []
        instance_num = 0

        with torch.no_grad():
            for i, (batch_meta) in enumerate(test_loader):
                pred, true = self._process_one_batch(batch_meta)
                batch_size = pred.shape[0]
                instance_num += batch_size
                pred = pred.reshape(-1)
                true = true.reshape(-1)
                batch_metric = np.array(metric(pred, true)) * batch_size
                batch_metrics_all.append(batch_metric)

        batch_metrics_all = np.stack(batch_metrics_all, axis=0)
        batch_metrics_all = batch_metrics_all.sum(axis=0) / instance_num

        # result save
        folder_path = './forecast_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        spcc, plcc = batch_metrics_all
        logger.info(f"In the _____test_____ function metrics ->: spcc:{spcc}, plcc:{plcc}")
        self.writer.add_scalar(f'{fold}/re_iqa/srcc_test_loss', spcc, global_step=0)
        self.writer.add_scalar(f'{fold}/re_iqa/plcc_test_loss', plcc, global_step=0)
        self.train_mode()

        return np.sum(batch_metrics_all)


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
    args.aug = False
    args.n_args = 1
    setting = "model_card_{}_dataset_card_{}_n_args_{}_epochs_{}".format(
        args.model_card, args.dataset_card, args.n_args, args.epochs
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
        exp.test(setting)
