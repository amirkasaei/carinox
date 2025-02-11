'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
from ruamel.yaml import YAML
import torch
import torch.nn as nn
import torch.nn.functional as F


import builtins
import os
# Save the original `open` function
original_open = open

def open_patched(file, *args, **kwargs):
    # Check if the path starts with the hardcoded prefix
    if file.startswith("/evaluation/rewards/content/BVQA/configs/"):
        # Remove the leading '/' and append the correct base directory
        relative_path = file.lstrip("/")
        file = os.path.join(r"E:/Reno_V/evaluation/rewards/", relative_path)

    if file.startswith("configs/"):
        # Remove the leading '/' and append the correct base directory
        relative_path = file.lstrip("/")
        file = os.path.join(r"E:/Reno_V/evaluation/rewards/content/BVQA/", relative_path)

    # Call the original `open` function with the (potentially modified) path
    return original_open(file, *args, **kwargs)

# Override the built-in `open` function
builtins.open = open_patched

from evaluation.rewards.content.BVQA.models.blip_vqa import blip_vqa
from rewards.reward_classes.base_reward import BaseRewardLoss
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from torchvision.transforms.functional import InterpolationMode
from PIL import Image


def clip_img_transform(size: int = 224):
    return Compose(
        [
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

class BVQAScore(BaseRewardLoss):
    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str
    ):
        self.device = device
        yaml = YAML(typ='rt')
        with open('evaluation/rewards/content/BVQA/configs/vqa.yaml', 'r') as f:
          self.config = yaml.load(f)

        self.bvqa_model = blip_vqa(pretrained=self.config['pretrained'], image_size=self.config['image_size'],
                       vit=self.config['vit'], vit_grad_ckpt=self.config['vit_grad_ckpt'], vit_ckpt_layer=self.config['vit_ckpt_layer'])

        self.bvqa_model = self.bvqa_model.to(self.device)
        self.bvqa_model.eval()

    def __call__(self, img_path: str, prompt: str) -> torch.Tensor:
        img = Image.open(img_path)
        preprocessed_img = clip_img_transform(480)(img)
        image = preprocessed_img.unsqueeze(0).to(self.device)

        return self.compute(image, prompt)

    def compute(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:

        bvqa_score = self.bvqa_model(image_features, text_features, train=False, inference="vqa_prob")[0]
        bvqa_score = float(''.join(map(str,  bvqa_score)))
        bvqa_loss = 1 - bvqa_score

        return bvqa_score, bvqa_loss
    
