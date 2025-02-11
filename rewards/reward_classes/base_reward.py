from abc import ABC, abstractmethod

import torch


class BaseRewardLoss(ABC):
    """
    Base class for reward functions implementing a differentiable reward function for optimization.
    """

    def __init__(self, name: str, weighting: float):
        self.name = name
        self.weighting = weighting

    @staticmethod
    def freeze_parameters(params: torch.nn.ParameterList):
        for param in params:
            param.requires_grad = False

    @abstractmethod
    def compute(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def __call__(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        pass
    
    # @abstractmethod
    # def into_cuda(self):
    #     pass

    # @abstractmethod
    # def into_cpu(self):
    #     pass