import inspect
from typing import Callable, List, Optional, Union

import torch
from diffusers import StableDiffusionPipeline

def freeze_params(params):
    for param in params:
        param.requires_grad = False

class RewardStableDiffusionV2(StableDiffusionPipeline):
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1", memsave=False):
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        super().__init__(
            vae=pipeline.vae.to("cuda"),
            text_encoder=pipeline.text_encoder.to("cuda"),
            tokenizer=pipeline.tokenizer,
            unet=pipeline.unet.to("cuda"),
            scheduler=pipeline.scheduler,
            safety_checker=None,  # disable safety checker for memory
            feature_extractor=pipeline.feature_extractor,
        )

        self.vae_scale_factor = 8

        if memsave:
            import memsave_torch.nn
            self.vae = memsave_torch.nn.convert_to_memory_saving(self.vae)
            self.unet = memsave_torch.nn.convert_to_memory_saving(self.unet)
            self.text_encoder = memsave_torch.nn.convert_to_memory_saving(self.text_encoder)

        # enable gradient checkpointing
        self.text_encoder.gradient_checkpointing_enable()
        self.unet.enable_gradient_checkpointing()
        self.vae.eval()
        self.text_encoder.eval()
        self.unet.eval()

        freeze_params(self.vae.parameters())
        freeze_params(self.unet.parameters())
        freeze_params(self.text_encoder.parameters())

        # If xformers is available, enable memory efficient attention
        try:
            self.unet.enable_xformers_memory_efficient_attention()
        except:
            pass

    @property
    def components(self):
        return {
            "vae": self.vae,
            "text_encoder": self.text_encoder,
            "tokenizer": self.tokenizer,
            "unet": self.unet,
            "scheduler": self.scheduler,
            "safety_checker": self.safety_checker,
            "feature_extractor": self.feature_extractor,
        }

    @property
    def do_classifier_free_guidance(self):
        # Keep CFG on for more dramatic changes
        return True

    def encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt=None):
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        text_inputs = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                     truncation=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_embeds = self.text_encoder(text_input_ids, return_dict=True)[0]

        uncond_inputs = self.tokenizer(negative_prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                       truncation=True, return_tensors="pt")
        uncond_input_ids = uncond_inputs.input_ids.to(device)
        negative_prompt_embeds = self.text_encoder(uncond_input_ids, return_dict=True)[0]

        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds, negative_prompt_embeds

    def decode_latents_tensors(self, latents):
        latents = latents / 0.18215
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def apply(self,
              latents: Optional[torch.Tensor] = None,
              prompt: Union[str, List[str]] = None,
              text_embeddings=None,
              height: Optional[int] = None,
              width: Optional[int] = None,
              timesteps: Optional[List[int]] = None,
              num_inference_steps: int = 50,  # More steps for refinement
              guidance_scale: float = 15.0,    # Increased guidance scale
              negative_prompt: Optional[Union[str, List[str]]] = None,
              num_images_per_prompt: Optional[int] = 1,
              eta: float = 0.0,
              generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
              callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
              callback_steps: Optional[int] = 1) -> torch.Tensor:

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        device = torch.device("cuda")

        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
            )

            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device=device,
                                                                timesteps=timesteps)

            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

            if latents is None:
                if isinstance(prompt, str):
                    batch_size = 1
                elif isinstance(prompt, list):
                    batch_size = len(prompt)
                else:
                    batch_size = prompt_embeds.shape[0] // 2 if self.do_classifier_free_guidance else prompt_embeds.shape[0]

                shape = (
                    batch_size * num_images_per_prompt,
                    self.unet.config.in_channels,
                    height // self.vae_scale_factor,
                    width // self.vae_scale_factor,
                )
                latents = torch.randn(shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
                latents = latents * self.scheduler.init_noise_sigma
            else:
                latents = latents.to(device)

            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            image = self.decode_latents_tensors(latents)
        return image


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps
