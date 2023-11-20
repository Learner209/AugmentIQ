'''
@File       :   ImageReward.py
@Time       :   2023/01/28 19:53:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   ImageReward Reward model.
* Based on CLIP code base and improved-aesthetic-predictor code base
* https://github.com/openai/CLIP
* https://github.com/christophschuhmann/improved-aesthetic-predictor
'''
import sys
import os

import torch
import torch.nn as nn
from PIL import Image
from ImageReward.models.BLIP.blip_pretrain import BLIP_Pretrain
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from loguru import logger
import torch.nn.functional as F

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1)
        )

        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (self.input_size + 1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)

    def forward(self, input):
        return self.layers(input)


class ImageReward(nn.Module):
    def __init__(self, ckpt_path, med_config, device='cpu'):
        super().__init__()
        self.device = device

        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config).to(device)
        self.mlp = MLP(768).to(device)

        assert os.path.isfile(ckpt_path), f"checkpoint file {ckpt_path} not found."
        assert os.path.exists(med_config), f"med_config file {med_config} not found."
        state_dict = torch.load(ckpt_path, map_location=device)
        self.load_state_dict(state_dict, strict=False)

        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072

        self.preprocess = None

        # freeze the weights of the pre-trained models
        for param in self.blip.parameters():
            param.requires_grad = False
        # make the MLP and projection layers differentiable
        for param in self.mlp.parameters():
            param.requires_grad = False
        for param in self.blip.vision_proj.parameters():
            param.requires_grad = False
        for param in self.blip.text_proj.parameters():
            param.requires_grad = False

    def forward(self, image_tensor, tokenized_prompt_input_ids, tokenized_prompt_input_attn_mask):
        """Judging the correlation score between images and prompts.

        Args:
            image_tensor (torch.Tensor): (batch_size X D) X C X H X W 
            tokenized_prompt_input_ids (torch.Tensor): bsz X D 
            tokenized_prompt_input_attn_mask (torch.Tensor): bsz X D 

        Returns:
            the ImageRewards score indicating the correlation score between images and prompts.
        """
        # logger.info(f"In the ImageRewards function, image_tensor.shape:{image_tensor.shape}, \
        #     tokenized_prompt_input_ids.shape:{tokenized_prompt_input_ids.shape}, \
        #     tokenized_prompt_input_attn_mask.shape:{tokenized_prompt_input_attn_mask.shape}")
        assert tokenized_prompt_input_ids.dim() == 2 and tokenized_prompt_input_attn_mask.dim() == 2

        image_embeds = self.blip.visual_encoder(image_tensor).to(image_tensor.device)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.float32).to(image_tensor.device)

        text_output = self.blip.text_encoder(tokenized_prompt_input_ids,
                                             attention_mask=tokenized_prompt_input_attn_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True,
                                             )

        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        # logger.info(f"rewards:{rewards}")

        return rewards

    def train_mode(self):
        for param in self.blip.parameters():
            param.requires_grad = False
        for param in self.mlp.parameters():
            param.requires_grad = True
        for param in self.blip.vision_proj.parameters():
            param.requires_grad = False
        for param in self.blip.text_proj.parameters():
            param.requires_grad = False

    def eval_mode(self):
        for param in self.blip.parameters():
            param.requires_grad = False
        for param in self.mlp.parameters():
            param.requires_grad = False
        for param in self.blip.vision_proj.parameters():
            param.requires_grad = False
        for param in self.blip.text_proj.parameters():
            param.requires_grad = False

    def score(self, prompt, image):
        # text encode
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)

        # image encode
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            if os.path.isfile(image):
                pil_image = Image.open(image)
        else:
            raise TypeError(r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.')

        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_embeds = self.blip.visual_encoder(image)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        text_output = self.blip.text_encoder(text_input.input_ids,
                                             attention_mask=text_input.attention_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True,
                                             )

        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        
        rewards = F.normalize(rewards, p=2, dim=1)
        # logger.info(f"rewards:{rewards}")

        return rewards


if __name__ == '__main__':
    # pip install image-reward
    # import ImageReward as RM
    model = ImageReward(med_config="augmentIQ/ImageReward/pretrained_model/med_config.json")

    rewards = model.score("<prompt>", "/home/liilu/Desktop/COURSE/Pic_proc/AIGC/BASELINE/augmentIQ/augmentIQ/sample_images/10004473376.jpg")
