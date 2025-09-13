import sys
# Add the project directory to sys.path to allow module imports
sys.path.append('/Compositional-Noise-Optimization')

import json
import logging
import os

import blobfile as bf
import torch
from datasets import load_dataset
from pytorch_lightning import seed_everything
from tqdm import tqdm

from arguments import parse_args
from models import get_model, get_multi_apply_fn
from rewards import get_reward_losses
from training import LatentNoiseTrainer, get_optimizer

import pandas as pd

from collections import defaultdict
def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)

def defaultdict_to_dict(d):
        if isinstance(d, defaultdict):
            d = {k: defaultdict_to_dict(v) for k, v in d.items()}
        return d

# Serialize to JSON and Save
def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def main(args):
    seed_everything(args.seed)
    bf.makedirs(f"{args.save_dir}/logs/")
    # Set up logging and name settings
    logger = logging.getLogger()
    
    settings = (
        f"{args.setting}{'_'}"
        f"{args.model}{'_' + args.prompt_file}"
        f"{'_' + ('adaptive' if not args.not_adaptive else 'normal')}"
        f"{'_k' + str(args.k)}"
        f"{'_no-optim' if args.no_optim else ''}_{args.seed}"
        f"_lr{args.lr}_gc{args.grad_clip}_iter{args.n_iters}"
        f"_reg{args.reg_weight if args.enable_reg else '0'}"
        f"{'_vqascore' + str(args.vqa_weight) if args.enable_vqa else ''}"
        f"{'_da_score' + str(args.da_weight) if args.enable_da else ''}"
        f"{'_imagereward' + str(args.imagereward_weight) if args.enable_imagereward else ''}"
        f"{'_hps' + str(args.hps_weight) if args.enable_hps else ''}"
    )
    file_stream = open(f"{args.save_dir}/logs/{settings}.txt", "w")
    handler = logging.StreamHandler(file_stream)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel("INFO")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logging.info(args)
    if args.device_id is not None:
        logging.info(f"Using CUDA device {args.device_id}")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = torch.device("cuda")
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    
    # Get reward losses
    reward_losses = get_reward_losses(args, dtype, device, args.cache_dir)

    # Get adabtive data
    categories, weights = None, None
    if not args.not_adaptive:
        categories = pd.read_json(f"assets/{args.category_file}.json")
        weights = pd.read_json(f"assets/{args.adaptive_weights_file}.json")

    # Get model and noise trainer
    pipe = get_model(args.model, dtype, device, args.cache_dir)

    trainer = LatentNoiseTrainer(
        reward_losses=reward_losses,
        model=pipe,
        n_iters=args.n_iters,
        n_inference_steps=args.n_inference_steps,
        seed=args.seed,
        save_all_images=args.save_all_images,
        save_every_10_image=args.save_every_10_image,
        save_every_5_image=args.save_every_5_image,
        adaptive=not args.not_adaptive,
        device=device,
        no_optim=args.no_optim,
        regularize=args.enable_reg,
        regularization_weight=args.reg_weight,
        grad_clip=args.grad_clip,
        categories=categories,
        weights=weights
    )

    # Create latents
    if args.model != "pixart":
        height = pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = pipe.unet.config.sample_size * pipe.vae_scale_factor
        shape = (
            1,
            pipe.unet.in_channels,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        )
    else:
        height = pipe.transformer.config.sample_size * pipe.vae_scale_factor
        width = pipe.transformer.config.sample_size * pipe.vae_scale_factor
        shape = (
            1,
            pipe.transformer.config.in_channels,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        )
    enable_grad = not args.no_optim
    if args.enable_multi_apply:
        multi_apply_fn = get_multi_apply_fn(
            model_type=args.multi_step_model,
            seed=args.seed,
            pipe=pipe,
            cache_dir=args.cache_dir,
            device=device,
            dtype=dtype,
        )
    else:
        multi_apply_fn = None

    fo = open(f"assets/{args.prompt_file}.txt", "r")
    prompts = fo.readlines()
    fo.close()

    seeds = torch.randint(0, 100, (int(args.k),))
    seeds[0] = 0

    # resulted seeds:
    # seeds = [0, 94, 58, 45, 15][:int(args.k)]
    
    holder = defaultdict(recursive_defaultdict)

    # Used seed (for prompt) instead of prompt (for seed) for better compatibility with ReNO's seed setting
    for n, seed in enumerate(seeds):    
        # seed = seed.item()
        seed_everything(seed)
        for i, prompt in tqdm(enumerate(prompts)):
            
            
            init_latents = torch.randn(shape, device=device, dtype=dtype)
            latents = torch.nn.Parameter(init_latents, requires_grad=enable_grad)
            optimizer = get_optimizer(args.optim, latents, args.lr, args.nesterov)

            prompt = prompt.strip()
            name = f"{i:03d}_{prompt[:150]}.png"
            save_dir = f"{args.save_dir}/{settings}/{name}"
            os.makedirs(save_dir, exist_ok=True)
            init_image, best_image, i30_best_image, init_rewards, best_rewards, i30_best_rewards = trainer.train(
                latents, prompt, optimizer, save_dir, multi_apply_fn, seed
            )
            
            holder[prompt][int(seed)]["best_loss"] = best_rewards["total"]
            holder[prompt][int(seed)]["best_i30_loss"] = i30_best_rewards["total"]
            holder[prompt][int(seed)]["initial_loss"] = init_rewards["total"]

            
            if n == 0 :
                holder[prompt]["min"]["best_loss"] = best_rewards["total"]
                holder[prompt]["min"]["best_i30_loss"] = i30_best_rewards["total"]
                holder[prompt]["min"]["initial_loss"] = init_rewards["total"]
                best_image.save(f"{save_dir}/best_image.png")

            else:
                if holder[prompt]["min"]["best_loss"] > best_rewards["total"]:
                    holder[prompt]["min"]["best_loss"] = best_rewards["total"]
                    holder[prompt]["min"]["initial_loss"] = init_rewards["total"]
                    best_image.save(f"{save_dir}/best_image.png")
                if holder[prompt]["min"]["best_i30_loss"] > i30_best_rewards["total"]:
                    holder[prompt]["min"]["best_i30_loss"] = i30_best_rewards["total"]
                    holder[prompt]["min"]["i30_initial_loss"] = init_rewards["total"]
                    i30_best_image.save(f"{save_dir}/i30_best_image.png")
            if i == 0:
                total_best_rewards = {k: 0.0 for k in best_rewards.keys()}
                total_init_rewards = {k: 0.0 for k in best_rewards.keys()}            
            
            for k in best_rewards.keys():
                total_best_rewards[k] += best_rewards[k]
                total_init_rewards[k] += init_rewards[k]
            best_image.save(f"{save_dir}/{seed}_best_image.png")
            init_image.save(f"{save_dir}/{seed}_init_image.png")
            i30_best_image.save(f"{save_dir}/{seed}_i30_best_image.png")
            
            logging.info(f"Initial rewards: {init_rewards}")
            logging.info(f"Best rewards: {best_rewards}")
        for k in total_best_rewards.keys():
            total_best_rewards[k] /= len(prompts)
            total_init_rewards[k] /= len(prompts)

        # save results to directory
        with open(f"{args.save_dir}/{settings}/results{seed}.txt", "w") as f:
            f.write(
                f"Mean initial all rewards: {total_init_rewards}\n"
                f"Mean best all rewards: {total_best_rewards}\n"
            )
        
        # log total rewards
        logging.info(f"Mean initial rewards: {total_init_rewards}")
        logging.info(f"Mean best rewards: {total_best_rewards}")
    regular_dict_holder = defaultdict_to_dict(holder)

    save_to_json(regular_dict_holder, f"{args.save_dir}/{settings}/holder.json")


if __name__ == "__main__":
    args = parse_args()
    main(args)
