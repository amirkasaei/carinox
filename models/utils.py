import logging
from typing import Any, Optional
import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LCMScheduler,
    Transformer2DModel,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from models.RewardPixart import RewardPixartPipeline, freeze_params
from models.RewardStableDiffusion import RewardStableDiffusion
from models.RewardStableDiffusionXL import RewardStableDiffusionXL


def get_model(
    model_name: str,
    dtype: torch.dtype,
    device: torch.device,
    cache_dir: str,
    enable_sequential_cpu_offload: bool = False,
    memsave: bool = False,
):
    logging.info(f"Loading model: {model_name}")
    if model_name == "sd-turbo":
        pipe = RewardStableDiffusion.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=dtype,
            variant="fp16",
            cache_dir=cache_dir,
            memsave=memsave,
        )
        pipe = pipe.to(device, dtype)
    elif model_name == "sdxl-turbo":
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
        )
        pipe = RewardStableDiffusionXL.from_pretrained(
            "stabilityai/sdxl-turbo",
            vae=vae,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
            cache_dir=cache_dir,
            memsave=memsave,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        pipe = pipe.to(device, dtype)
    elif model_name == "pixart":
        pipe = RewardPixartPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            torch_dtype=dtype,
            cache_dir=cache_dir,
            memsave=memsave,
        )
        pipe.transformer = Transformer2DModel.from_pretrained(
            "PixArt-alpha/PixArt-Alpha-DMD-XL-2-512x512",
            subfolder="transformer",
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )
        pipe.scheduler = DDPMScheduler.from_pretrained(
            "PixArt-alpha/PixArt-Alpha-DMD-XL-2-512x512",
            subfolder="scheduler",
            cache_dir=cache_dir,
        )

        # speed-up T5
        pipe.text_encoder.to_bettertransformer()
        pipe.transformer.eval()
        freeze_params(pipe.transformer.parameters())
        pipe.transformer.enable_gradient_checkpointing()
        pipe = pipe.to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    if enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    return pipe


def get_multi_apply_fn(
    model_type: str,
    seed: int,
    pipe: Optional[Any] = None,
    cache_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    generator = torch.Generator("cuda").manual_seed(seed)
    if model_type == "flux":
        return lambda latents, prompt: torch.no_grad(pipe.apply)(
            latents=latents,
            prompt=prompt,
            num_inference_steps=4,
            generator=generator,
        )
    elif model_type == "sdxl":
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
        )
        pipe = RewardStableDiffusionXL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            vae=vae,
            use_safetensors=True,
            cache_dir=cache_dir,
        )
        pipe = pipe.to(device, dtype)
        pipe.enable_sequential_cpu_offload()
        return lambda latents, prompt: torch.no_grad(pipe.apply)(
            latents=latents,
            prompt=prompt,
            guidance_scale=5.0,
            num_inference_steps=50,
            generator=generator,
        )
    elif model_type == "sd2":
        sd2_base = "stabilityai/stable-diffusion-2-1-base"
        scheduler = EulerDiscreteScheduler.from_pretrained(
            sd2_base,
            subfolder="scheduler",
            cache_dir=cache_dir,
        )
        pipe = RewardStableDiffusion.from_pretrained(
            sd2_base,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            scheduler=scheduler,
        )
        pipe = pipe.to(device, dtype)
        pipe.enable_sequential_cpu_offload()
        return lambda latents, prompt: torch.no_grad(pipe.apply)(
            latents=latents,
            prompt=prompt,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
