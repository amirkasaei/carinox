import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .BLIP.models.blip_vqa import blip_vqa
from rewards.reward_classes.VQA_reward import VQARewardLoss
from rewards.reward_classes.base_reward import BaseRewardLoss
"""

paper's own custom implementation of the ranking is used for overwriting blipvqa_model's rank_answers method

"""
from typing import Tuple
import torch
import pandas as pd
import os.path

DA_GENERATION_PATH= 'assets/DA_questions.json'


class DAScoreLoss(VQARewardLoss):
    """DA score loss for optimization."""

    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str,
        memsave: bool = False,
    ):
        self.dtype = dtype
        self.questions = None
        self.device  = device

        url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'
        self.model = blip_vqa(pretrained=url, image_size=480, vit='base') # set image size to 480
        self.model.eval()
        self.model = self.model.to(device = self.device, dtype = self.dtype)

        self.blipvqa_model = self.model
    
        # Freeze
        BaseRewardLoss.freeze_parameters(self.blipvqa_model.parameters())

        super().__init__("DAscore", weighting)

        self.device = device

    def get_questions(self, prompt: str, mode = "FILE"):
        if self.questions is None :
            file_path = DA_GENERATION_PATH
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    self.questions = pd.read_json(f, lines=True)
            else:
                raise Exception("mount file")

        results = self.questions[self.questions["prompt"] == prompt]

        # Check if the result is empty
        if results.empty:
            raise Exception(f"No entries found for p={prompt}" )
        try:
            # Attempt to access the first elements of 'questions' and 'parsed_input'
            return results["questions"].iloc[0], results["parsed_input"].iloc[0]
        except KeyError as e:
            raise KeyError(f"KeyError encountered: {e}. Possible missing keys in DataFrame.")

    def evaluate_question_image(self, image: torch.Tensor, question: str) -> Tuple[float, float]:
        vqa_pred = self.blipvqa_model(
            image,
            question,
            train = False,
            inference="rank",
            answer=['yes','no'],
            n = 2
        )
        pos_score, neg_score = vqa_pred[1][0][0], vqa_pred[1][0][1]
        return pos_score, neg_score
    
    def compute(
        self, image: torch.Tensor, text_input: torch.Tensor, use_neg_scores: bool = False, neg_score_coef: float = 1.0
    ) -> torch.Tensor:
        questions, parsed_input = self.get_questions(text_input)

        vqa_scores = []

        # Initialize yes/no scores
        pos_scores = []
        neg_scores = []

        # Iterate through questions and compute VQA scores
        for question in questions:
            pos_score, neg_score = self.evaluate_question_image(image, question)
            if not use_neg_scores:
                # This sets neg_score = 0 but does not require grad. That’s usually fine,
                # because it's effectively "constant = 0." 
                # If you do need it to be part of the graph for some reason, see note below.
                neg_score = neg_score.new_zeros([])  
            pos_scores.append(pos_score)
            neg_scores.append(neg_score)
        # Instead of torch.tensor(...), do:
        pos_scores = torch.stack(pos_scores)  # shape = [num_questions]
        neg_scores = torch.stack(neg_scores)  # shape = [num_questions]

        # Now the gradient connections are preserved:
        diff_scores = pos_scores - neg_score_coef * neg_scores
        vqa_scores.append(diff_scores)

        # Compute final DA-score as average of the individual assertion alignment scores
        da_score = torch.mean(torch.stack(vqa_scores), dim=-1)[0] # output as a float not list
        da_loss = 1 - da_score

        return da_loss, da_score
    