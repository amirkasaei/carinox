from abc import ABC, abstractmethod

import torch
from .base_reward import BaseRewardLoss

class VQARewardLoss(BaseRewardLoss):
    """
    Abstract base class for reward functions based on Visual Question Answering (VQA).
    """

    def __init__(self, name: str, weighting: float):
        super().__init__(name, weighting)

    @abstractmethod
    def get_questions(self, prompt: str):
        pass

    @abstractmethod
    def evaluate_question_image(
        self, image_features: torch.Tensor, question_features: torch.Tensor
    ) -> torch.Tensor:
        pass

    def __call__(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        return self.compute(image, prompt)
