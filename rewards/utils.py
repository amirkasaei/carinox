from typing import Any, List
import json 

import torch
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, Resize)
from transformers import AutoProcessor

from rewards.reward_classes.base_reward import BaseRewardLoss
from rewards.hps import HPSLoss
from rewards.imagereward import ImageRewardLoss
from rewards.VQAscore import VQA
from rewards.DAscore import DAScoreLoss


def get_reward_losses(
    args: Any, dtype: torch.dtype, device: torch.device, cache_dir: str
) -> List[BaseRewardLoss]:
    embedding_losses, vqa_loss, da_loss = [], [], []
    if args.enable_hps:
        embedding_losses.append(
            HPSLoss(args.hps_weight, dtype, device, cache_dir)
        )
    if args.enable_imagereward:
        embedding_losses.append(
            ImageRewardLoss(args.imagereward_weight, dtype, device, cache_dir)
        )
    if args.enable_vqa:
        vqa_loss.append(
            VQA(args.vqa_weight, dtype, device, cache_dir)
        )
    if args.enable_da:
        da_loss.append(
            DAScoreLoss(args.da_weight, dtype, device, cache_dir)
        ) 
    
    reward_losses= {'vqa': vqa_loss, 'embedding': embedding_losses, "da": da_loss} 
    return reward_losses

type_size_mapping = {'embedding':224, 'vqa':None, 'da':480}

def clip_img_transform(type, image):
    size = type_size_mapping[type]
    if not size:
        return image
    transform = Compose(
        [
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    return transform(image)