from abc import ABC, abstractmethod

import torch
from .base_reward import BaseRewardLoss

class EmbeddingRewardLoss(BaseRewardLoss):
    """
    Abstract base class for reward functions based on image and text embeddings.
    """

    def __init__(self, name: str, weighting: float):
        super().__init__(name, weighting)

    # Retain the abstract methods from BaseRewardLoss
    @abstractmethod
    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_text_features(self, prompt: str) -> torch.Tensor:
        pass

    def process_features(self, features: torch.Tensor) -> torch.Tensor:
        features_normed = features / features.norm(dim=-1, keepdim=True)
        return features_normed
    
    def __call__(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        image_features = self.get_image_features(image)
        text_features = self.get_text_features(prompt)

        image_features_normed = self.process_features(image_features)
        text_features_normed = self.process_features(text_features)

        loss, score = self.compute(image_features_normed, text_features_normed)
        return loss, score
