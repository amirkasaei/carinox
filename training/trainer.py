import logging
import math
from typing import Dict, List, Optional, Tuple

import PIL
import PIL.Image
import torch
from diffusers import DiffusionPipeline

from rewards import clip_img_transform
from rewards.reward_classes.base_reward import BaseRewardLoss

import pandas as pd

class LatentNoiseTrainer:
    """Trainer for optimizing latents with reward losses."""

    def __init__(
        self,
        reward_losses: List[BaseRewardLoss],
        model: DiffusionPipeline,
        n_iters: int,
        n_inference_steps: int,
        seed: int,
        no_optim: bool = False,
        regularize: bool = True,
        regularization_weight: float = 0.01,
        grad_clip: float = 0.1,
        save_all_images: bool = False,
        save_every_10_image: bool = False,
        save_every_5_image: bool = False,
        device: torch.device = torch.device("cuda"),
    ):
      
        self.reward_losses = reward_losses
        self.model = model
        self.n_iters = n_iters
        self.n_inference_steps = n_inference_steps
        self.seed = seed
        self.no_optim = no_optim
        self.regularize = regularize
        self.regularization_weight = regularization_weight
        self.save_all_images = save_all_images
        self.save_every_10_image = save_every_10_image
        self.save_every_5_image = save_every_5_image
        self.device = device
        self.grad_clip = grad_clip

    def train(
        self,
        latents: torch.Tensor,
        prompt: str,
        optimizer: torch.optim.Optimizer,
        save_dir: Optional[str] = None,
        multi_apply_fn=None,
        s = 0,
    ) -> Tuple[PIL.Image.Image, Dict[str, float], Dict[str, float]]:
        logging.info(f"Optimizing latents for prompt '{prompt}'.")
        best_loss = torch.inf
        best_image = None
        initial_image = None
        initial_rewards = None
        best_rewards = None
        best_latents = None
        latent_dim = math.prod(latents.shape[1:])

        i30_best_loss = None
        i30_best_image = None
        i30_best_rewards = None
        i30_best_latents = None

        for iteration in range(self.n_iters):
            to_log = ""
            rewards = {}
            optimizer.zero_grad()
            generator = torch.Generator("cuda").manual_seed(s)
            image = self.model.apply(
                latents=latents,
                prompt=prompt,
                generator=generator,
                num_inference_steps=self.n_inference_steps,
            )
            if initial_image is None:
                if multi_apply_fn is not None:
                    initial_image = multi_apply_fn(latents.detach(), prompt)
                else:
                    initial_image = image
                image_numpy = (
                    initial_image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
                )
                initial_image = DiffusionPipeline.numpy_to_pil(image_numpy)[0]
            if self.no_optim:
                best_image = image
                break

            total_loss = 0
            grad_clones = []

            if self.regularize:
                # compute in fp32 to avoid overflow
                latent_norm = torch.linalg.vector_norm(latents).to(torch.float32)
                log_norm = torch.log(latent_norm)
                regularization = self.regularization_weight * (
                    0.5 * latent_norm**2 - (latent_dim - 1) * log_norm
                )

            for reward_type, metrics in self.reward_losses.items():
                preprocessed_image = clip_img_transform(reward_type, image)
                for reward_loss in metrics:
                    # print(f"{reward_loss.name}:{reward_loss.weighting}")
                    
                    loss, _ = reward_loss(preprocessed_image, prompt)
                    to_log += f"{reward_loss.name}: {loss.item():.4f}, "
                    rewards[reward_loss.name] = loss.item()

                    total_loss += loss.item()
                    loss *= reward_loss.weighting 
                    if self.regularize:
                        loss += regularization.to(loss.dtype)

                    loss.backward(retain_graph=True)

                    gradient_norm = torch.nn.utils.clip_grad_norm_(latents, self.grad_clip)
                    to_log += f"{reward_loss.name} grad norm: {gradient_norm}, "

                    grad_clones.append(latents.grad.clone())
                    optimizer.zero_grad()


            rewards["total"] = total_loss
            to_log += f"Total: {total_loss:.4f}"
            total_reward_loss = total_loss
        

            
            
            to_log += f", Latent norm: {latent_norm.item()}, "
    
            if total_reward_loss < best_loss:
                if iteration < 30 : 
                    i30_best_loss = total_reward_loss
                    i30_best_image = image
                    i30_best_rewards = rewards
                    i30_best_latents = latents.detach().cpu()    
                best_loss = total_reward_loss
                best_image = image
                best_rewards = rewards
                best_latents = latents.detach().cpu()
            if iteration != self.n_iters - 1:
                latents.grad = sum(grad_clones)
                gradient_norm = torch.nn.utils.clip_grad_norm_(latents, self.grad_clip)
                to_log += f"latent grad norm: {gradient_norm}"
                optimizer.step()
                optimizer.zero_grad()
            
            if self.save_all_images:
                image_numpy = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
                image_pil = DiffusionPipeline.numpy_to_pil(image_numpy)[0]
                image_pil.save(f"{save_dir}/{s}_{iteration}.png")
            if self.save_every_10_image and iteration % 10 == 0: 
                image_numpy = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
                image_pil = DiffusionPipeline.numpy_to_pil(image_numpy)[0]
                image_pil.save(f"{save_dir}/{s}_{iteration}.png")
            if self.save_every_5_image and iteration % 5 == 0: 
                image_numpy = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
                image_pil = DiffusionPipeline.numpy_to_pil(image_numpy)[0]
                image_pil.save(f"{save_dir}/{s}_{iteration}.png")

            logging.info(f"Iteration {iteration}: {to_log}")
            if initial_rewards is None:
                initial_rewards = rewards
        image_numpy = best_image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        best_image_pil = DiffusionPipeline.numpy_to_pil(image_numpy)[0]

        i30_image_numpy = i30_best_image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        i30_best_image_pil = DiffusionPipeline.numpy_to_pil(i30_image_numpy)[0]

        return initial_image, best_image_pil, i30_best_image_pil, initial_rewards, best_rewards, i30_best_rewards