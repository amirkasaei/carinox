import torch
import os
import pandas as pd
from evaluation.rewards.BVQAscore import BVQAScore

class Evaluator:
    def __init__(self, model_name, cache_dir, weighting, eval_dir, save_dir):
        self.model_name = model_name
        self.weighting = weighting
        self.cache_dir = cache_dir
        self.eval_dir = eval_dir
        self.save_dir = save_dir
        self.dtype = torch.float16
        self.device = torch.device("cuda")

    def evaluate(self):
        model_score = None
        if self.model_name=='bvqa':
            model_score = BVQAScore(
                        self.weighting,
                        self.dtype,
                        self.device,
                        self.cache_dir)
            
        prompts = []
        scores = []

        for folder in sorted(os.listdir(self.eval_dir)):
            folder_path = os.path.join(self.eval_dir, folder)
            
            if os.path.isdir(folder_path) and '_' in folder:
                prompt = folder.split('_', 1)[1]
                img_path = os.path.join(folder_path, "best_image.png")

                prompts.append(prompt)
                scores.append(model_score(img_path, prompt)[0])
                
            
        
        df = pd.DataFrame({"Name": prompts, f"{self.model_name}_score": scores})

        output_dir = f"{self.eval_dir}/{self.model_name}_evaluation_scores.csv"
        df.to_csv(output_dir)
        return f"Results saved to {output_dir}."
