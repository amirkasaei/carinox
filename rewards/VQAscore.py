from rewards.reward_classes.VQA_reward import VQARewardLoss

import torch
from .t2v_metrics.vqascore import VQAScore

import gc
gc.collect()
torch.cuda.empty_cache()

class VQA(VQARewardLoss):

    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str,
        memsave: bool = False,
    ):

        self.name = "VQAScore"
        self.weighting = weighting
        self.dtype = dtype
        self.device = device

        # Using 'xl' instead of 'xxl' to save VRAM
        self.model = VQAScore(model='clip-flant5-xl', device=self.device)
        self.model.model.model.requires_grad_(False)

    def get_questions(self, prompt):
        return super().get_questions(prompt)
    
    def evaluate_question_image(self, image_features, question_features):
        return super().evaluate_question_image(image_features, question_features)
    
    def compute(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        vqa_score = self.model(images=[image_features], texts=[text_features])[0][0] # only use the first pair
        vqa_loss = 1 - vqa_score

        return vqa_loss, vqa_score