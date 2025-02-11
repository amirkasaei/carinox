import torch
from transformers import AutoModel

from rewards.reward_classes.embedding_reward import EmbeddingRewardLoss


class PickScoreLoss(EmbeddingRewardLoss):
    """PickScore reward loss function for optimization."""

    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str,
        tokenizer,
        memsave: bool = False,
    ):
        self.tokenizer = tokenizer
        self.model = AutoModel.from_pretrained(
            "yuvalkirstain/PickScore_v1", cache_dir=cache_dir
        ).eval()
        if memsave:
            import memsave_torch.nn

            self.model = memsave_torch.nn.convert_to_memory_saving(
                self.model
            )
        self.model = self.model.to(device, dtype=dtype)
        self.freeze_parameters(self.model.parameters())
        super().__init__("PickScore", weighting)
        self.model._set_gradient_checkpointing(True)

    def get_image_features(self, image) -> torch.Tensor:
        reward_img_features = self.model.get_image_features(image)
        return reward_img_features

    def get_text_features(self, prompt: str) -> torch.Tensor:
        prompt_token = self.tokenizer(
            prompt, return_tensors="pt", padding=True, max_length=77, truncation=True
        ).to("cuda")
        reward_text_features = self.model.get_text_features(**prompt_token)
        return reward_text_features

    def compute(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        pickscore_score = ((image_features @ text_features.T)).mean()
        pickscore_loss = 1 - pickscore_score
        
        return pickscore_loss, pickscore_score
