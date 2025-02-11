import torch
from transformers import CLIPModel

from rewards.reward_classes.embedding_reward import EmbeddingRewardLoss


class CLIPLoss(EmbeddingRewardLoss):
    """CLIP reward loss function for optimization."""

    def __init__(
        self,
        weigthing: float,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str,
        tokenizer,
        memsave: bool = False,
    ):
        self.tokenizer = tokenizer
        self.model = CLIPModel.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            cache_dir=cache_dir,
        )
        # freeze all models parameters
        if memsave:
            import memsave_torch.nn

            self.model = memsave_torch.nn.convert_to_memory_saving(self.model)
        self.model = self.model.to(device, dtype=dtype)
        self.model.eval()
        self.freeze_parameters(self.model.parameters())
        super().__init__("CLIP", weigthing)
        self.model.gradient_checkpointing_enable()

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        clip_img_features = self.model.get_image_features(image)
        return clip_img_features

    def get_text_features(self, prompt: str) -> torch.Tensor:
        prompt_token = self.tokenizer(
            prompt, return_tensors="pt", padding=True, max_length=77, truncation=True
        ).to("cuda")
        clip_text_features = self.model.get_text_features(**prompt_token)
        return clip_text_features

    def compute(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        clip_score =  (image_features @ text_features.T).mean()
        clip_loss = 1 - clip_score

        return clip_loss, clip_score
