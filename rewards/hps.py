import huggingface_hub
import torch
from hpsv2.src.open_clip import create_model, get_tokenizer

from rewards.reward_classes.embedding_reward import EmbeddingRewardLoss

class HPSLoss(EmbeddingRewardLoss):
    """HPS reward loss function for optimization."""

    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str,
        memsave: bool = False,
    ):
        self.model = create_model(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision=dtype,
            device=device,
            cache_dir=cache_dir,
        )
        self.device= device
        self.dtype= dtype

        checkpoint_path = huggingface_hub.hf_hub_download(
            "xswu/HPSv2", "HPS_v2.1_compressed.pt", cache_dir=cache_dir
        )
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)["state_dict"]
        )
        self.hps_tokenizer = get_tokenizer("ViT-H-14")
        if memsave:
            import memsave_torch.nn

            self.model = memsave_torch.nn.convert_to_memory_saving(self.model)
        self.model = self.model.to(device, dtype=dtype)
        self.model.eval()
        self.freeze_parameters(self.model.parameters())
        super().__init__("HPS", weighting)
        self.model.set_grad_checkpointing(True)

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        hps_image_features = self.model.encode_image(image)
        return hps_image_features

    def get_text_features(self, prompt: str) -> torch.Tensor:
        hps_text = self.hps_tokenizer(prompt).to("cuda")
        hps_text_features = self.model.encode_text(hps_text)
        return hps_text_features

    def compute(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        logits_per_image = image_features @ text_features.T
        hps_score = torch.diagonal(logits_per_image)[0]
        hps_loss = 1 - hps_score

        return hps_loss, hps_score
