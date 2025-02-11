from typing import Any, List
import json 

import torch
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, Resize)
from transformers import AutoProcessor

from rewards.aesthetic import AestheticLoss
from rewards.reward_classes.base_reward import BaseRewardLoss
from rewards.clip import CLIPLoss
from rewards.hps import HPSLoss
from rewards.imagereward import ImageRewardLoss
from rewards.pickscore import PickScoreLoss
from rewards.VQAscore import VQA
from rewards.DAscore import DAScoreLoss


def get_reward_losses(
    args: Any, dtype: torch.dtype, device: torch.device, cache_dir: str
) -> List[BaseRewardLoss]:
    if args.enable_clip or args.enable_pickscore:
        tokenizer = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=cache_dir
        )
    reno_losses = []
    if args.enable_hps:
        reno_losses.append(
            HPSLoss(args.hps_weighting, dtype, device, cache_dir, memsave=args.memsave)
        )
    if args.enable_imagereward:
        reno_losses.append(
            ImageRewardLoss(
                args.imagereward_weighting,
                dtype,
                device,
                cache_dir,
                memsave=args.memsave,
            )
        )
    if args.enable_clip:
        reno_losses.append(
            CLIPLoss(
                args.clip_weighting,
                dtype,
                device,
                cache_dir,
                tokenizer,
                memsave=args.memsave,
            )
        )

    if args.enable_pickscore:
        reno_losses.append(
            PickScoreLoss(
                args.pickscore_weighting,
                dtype,
                device,
                cache_dir,
                tokenizer,
                memsave=args.memsave,
            )
        )
    
    if args.enable_aesthetic:
        reno_losses.append(
            AestheticLoss(
                args.aesthetic_weighting, dtype, device, cache_dir, memsave=args.memsave
            )
        )

    vqa_losses = []
    da_losses = []
    if args.enable_vqa_score:
        vqa_losses.append(
            VQA(
                args.vqa_score_weighting, dtype, device, cache_dir, memsave=args.memsave
            )
        )
    if args.enable_da_score:
        da_losses.append(
            DAScoreLoss(
                args.da_score_weighting, dtype, device, cache_dir, memsave=args.memsave
            )
        ) 

    reward_losses= { 'vqa': vqa_losses,'reno': reno_losses, "da": da_losses}#,'diff_env': diff_env_losses}    
    return reward_losses



def send_command(process, command):
    process.stdin.write(json.dumps(command) + "\n")
    process.stdin.flush()

    while True:
        response = process.stdout.readline().strip()
        if response: 
            print("Received response:", response)
            try:
                response_data = json.loads(response)
                if response_data.get("status") == "initialized":
                    print("Model successfully initialized.")
                    break 
                elif response_data.get("error"):
                    raise Exception(f"Error from subprocess: {response_data['error']}")
            except json.JSONDecodeError:
                print(f"Invalid response: {response}")
            except Exception as e:
                print(f"Error processing response: {e}")
                break  

type_size_mapping = {'reno':224, 'vqa':None, 'da':480}

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