import os
import pandas as pd
import json

image_data_path = 'E:/train_sample.json'
main_path = 'E:\ReNO_V\Generation\example-prompts'
file_name = 'bvqa_evaluation_scores.csv'
output = 'E:\ReNO_V\Generation'    

image_data_path = 'E:/train_sample.json'

with open(image_data_path, 'r', encoding='utf-8') as inputfile:
    data = []
    for line in inputfile:
        data.append(json.loads(line.strip())) 

categories_df = pd.DataFrame(data)

final_df = pd.DataFrame({'category': categories_df['category'].unique()})
new_row = pd.DataFrame({'category': ['Overall']})
final_df = pd.concat([final_df, new_row], ignore_index=True)

methods = []
scores =[]

for folder in sorted(os.listdir(main_path)):
    folder_path = os.path.join(main_path, folder)
    
    if os.path.isdir(folder_path):
        bvqa_score_file = os.path.join(folder_path, file_name)
        scores_df = pd.read_csv(bvqa_score_file)  
        temp = []
        scores_df['Name'] = scores_df['Name'].str.replace('.png', '', regex=False)
        merged_df = pd.merge(scores_df, categories_df[['prompt', 'category']], left_on='Name', right_on='prompt', how='left')

        if len(merged_df) != 200:
            print(f"Error: Merged DataFrame does not have 200 rows. It has {len(merged_df)} rows.")
            continue

        category_means = merged_df.groupby("category")["bvqa_score"].mean().reset_index()
        temp.extend(category_means['bvqa_score'])
        overall_mean = merged_df["bvqa_score"].mean()
        temp.append(overall_mean)

        final_df[folder]= temp

output_dir = f"{output}/bvqa_scores.csv"
final_df.to_csv(output_dir)
print(f"Results saved to {output_dir}.")