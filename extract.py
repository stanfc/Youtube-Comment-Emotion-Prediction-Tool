import json
import random
from tqdm import tqdm

def extract_keywords(text):
    return text.split() 

train_texts = []  
train_labels = []  

with open("trainword.json", encoding='utf-8') as f:
    data = json.load(f)

for i in tqdm(data, desc="Processing training data"):
    for j in i:
        x = random.random()
        if x < 0.0035:
            strs = extract_keywords(j[0])
            str = ','.join(strs)
            train_texts.append(str)
            train_labels.append(int(j[1]))
output_data = {
    "train_texts": train_texts,
    "train_labels": train_labels
}

with open("output_train_data.json", "w", encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
