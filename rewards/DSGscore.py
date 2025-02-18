from base_reward import BaseRewardLoss
from VQA_reward import VQAReward
import torch
import json
from content.DSG.dsg.query_utils import generate_dsg
from content.DSG.dsg.query_utils_presaved import generate_dsg_presaved
from content.DSG.dsg.parse_utils import parse_tuple_output, parse_dependency_output, parse_question_output
from content.DSG.dsg.vqa_utils import MPLUG
import pandas as pd
from PIL import Image
import openai
from content.DSG.dsg.openai_utils import openai_setup, openai_completion

# class DSG(VQAReward):
#     def __init__(
#         self,
#         weighting: float,
#         dtype: torch.dtype,
#         device: torch.device,
#         cache_dir: str
#     ):

#         self.vqa_model = MPLUG()

#     def __call__(self, img_path: str, prompt: str) -> torch.Tensor:
#         img = Image.open(img_path)
#         return self.compute(img, prompt)

#     def get_questions(self, prompt):
#         id2prompts ={'custom_id': {'input':prompt,}}
#         id2tuple_outputs, id2question_outputs, id2dependency_outputs = generate_dsg(id2prompts,generate_fn=openai_completion)

#         qid2tuple = parse_tuple_output(id2tuple_outputs['custom_id']['output'])
#         qid2dependency = parse_dependency_output(id2dependency_outputs['custom_id']['output'])
#         qid2question = parse_question_output(id2question_outputs['custom_id']['output'])

#         return qid2tuple, qid2dependency, qid2question

#     def evaluate_question_image(
#         self, image_features: torch.Tensor, question_features: torch.Tensor
#     ) -> torch.Tensor:
#         return self.vqa_model.vqa(image_features, question_features)

#     def compute(
#         self, image_features: torch.Tensor, text_features: torch.Tensor
#     ) -> torch.Tensor:

#         qid2tuple, qid2dependency, qid2question = self.get_questions(text_features)

#         qid2answer = {}
#         qid2scores = {}
#         qid2validity = {}

#         for id, question in qid2question.items():
#             answer = self.evaluate_question_image(image_features, question)
#             qid2answer[id] = answer
#             qid2scores[id] = float(answer == 'yes')

#         # zero-out scores from invalid questions
		
#         for id, parent_ids in qid2dependency.items():
#             # zero-out scores if parent questions are answered 'no'
#             any_parent_answered_no = False
#             for parent_id in parent_ids:
#                 if parent_id == 0:
#                     continue
#                 if qid2scores[parent_id] == 0:
#                     any_parent_answered_no = True
#                     break
#             if any_parent_answered_no:
#                 qid2scores[id] = 0
#                 qid2validity[id] = False
#             else:
#                 qid2validity[id] = True

#         DSG_score = sum(qid2scores.values()) / len(qid2scores)
#         DSG_loss = 1 - DSG_score

#         return DSG_score, DSG_loss

class DSG(BaseRewardLoss):
	def __init__(
		self,
		weighting: float,
		dtype: torch.dtype,
		device: torch.device,
		cache_dir: str,
		memsave: bool
	):


		self.vqa_model = MPLUG()
		# self.vqa_model= InstructBLIP()
		self.device = device  # Store the device
		self.dtype = dtype  # Store the dtype
		self.memsave= memsave
		self.weighting= weighting
		self.cache_dir= cache_dir
		self.study_prompts, self.train_prompts = extract_prompts()


	def get_image_features(self, image) -> torch.Tensor:
		pass

	def get_text_features(self, prompt: str) -> torch.Tensor:
		pass

	def compute_loss(self, image_features, text_features):
		return super().compute_loss(image_features, text_features)
	
	def __call__(self, img_path: str, prompt: str) -> torch.Tensor:
		return self.compute(img_path, prompt)

	def compute(
		self, image_features: torch.Tensor, text_features: torch.Tensor
	) -> torch.Tensor:
		

		id2prompts ={}
		for i ,prompt in enumerate(self.study_prompts):
			id2prompts['custom_'+str(i)]= {'input': prompt,}
		for i ,prompt in enumerate(self.train_prompts):
			id2prompts['train_custom_'+str(i)]= {'input': prompt,}
		id2tuple_outputs, id2question_outputs, id2dependency_outputs= generate_dsg_presaved(id2prompts,generate_fn=openai_completion)
	
		prompt=text_features.lower()
		prompt= prompt.strip()
		if prompt[-1]=='.':
			prompt=prompt[:-1]
			prompt.strip()

		if prompt in self.study_prompts:
			indx= self.study_prompts.index(prompt)
			keynum = 'custom_'+str(indx)
			
		elif prompt in self.train_prompts:
			indx= self.train_prompts.index(prompt)
			# print(indx)
			# print(prompt)
			keynum = 'train_custom_'+str(indx)
			
		
		qid2tuple = parse_tuple_output(id2tuple_outputs[keynum]['output'])
		qid2dependency = parse_dependency_output(id2dependency_outputs[keynum]['output'])
		qid2question = parse_question_output(id2question_outputs[keynum]['output'])
		
		qid2answer = {}
		qid2scores = {}
		qid2validity = {}


		# print(PATH+image_features)
		img = Image.open(image_features)
		# print('after image')
		for id, question in qid2question.items():
			answer = self.vqa_model.vqa(img, question)
			qid2answer[id] = answer
			qid2scores[id] = float(answer == 'yes')
			#3) zero-out scores from invalid questions

		for id, parent_ids in qid2dependency.items():
			# zero-out scores if parent questions are answered 'no'
			any_parent_answered_no = False
			for parent_id in parent_ids:
				if parent_id == 0:
					continue
				if qid2scores[parent_id] == 0:
					any_parent_answered_no = True
					break
			if any_parent_answered_no:
				qid2scores[id] = 0
				qid2validity[id] = False
			else:
				qid2validity[id] = True

		DSG_score = sum(qid2scores.values()) / len(qid2scores)
		DSG_loss = 1 - DSG_score

		return DSG_loss


def extract_last_input(line):
	"""
	Extracts the text after the last 'input:' in a given line.
	"""
	# Split the line into parts by 'input:'
	parts = line.split("input:")
	# Return the last part after stripping whitespace
	return parts[-1].strip() if len(parts) > 1 else None

def extract_prompts():
	new_prompts= []
	# Read the file and process each line
	with open("rewards/content/DSG/assets/prompts.txt", "r") as file:
		for line in file:
			extracted_input = extract_last_input(line)
			if extracted_input:
				new_prompts.append( extracted_input.split('\\noutput:')[0])

	train_prompts= []
	with open("rewards/content/DSG/assets/train_prompts.json", "r") as file:
		train_prompts= json.load(file)
	
	return new_prompts, train_prompts

import sys

if __name__ == "__main__":

	model = None

	while True:
		# Read a command from stdin
		line = sys.stdin.readline()
		if not line:
			break  # Exit if no more input
		
		try:
			command = json.loads(line.strip())
			
			if command["action"] == "init":
				# Initialize the model
				params= command["param"]
				
				if params['dtype']=='torch.float16':
					dtype= torch.float16
				else:
					dtype= torch.float32
					
				model = DSG(
					weighting= params['dsg_score_weighting'], 
					dtype = dtype, 
					device= torch.device(params['device']), 
					cache_dir= params['cache_dir'], 
					memsave=params["memsave"]
				)
				
				# sys.stdout.write(json.dumps({"status": "initialized"}) + "\n")
				print(json.dumps({"status": "initialized"}))
				sys.stdout.flush()

			elif command["action"] == "call" and model:
				# Call the model's __call__ method
				input_data = command["input_data"]
				result = model(input_data["image"], input_data["prompt"])
				print(json.dumps({"result": result, "name": "DSGScore", 'weighting': model.weighting}))
				sys.stdout.flush()

			elif command["action"] == "exit":
				print(json.dumps({"status": "exiting"}))
				sys.stdout.flush()
				break

			else:
				print(json.dumps({"error": "Invalid command"}))
				sys.stdout.flush()

		except Exception as e:
			print(json.dumps({"error": str(e)}))
			sys.stdout.flush()
