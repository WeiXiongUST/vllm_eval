import numpy as np
import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")

ds = load_dataset("json", data_files="/home/wx/exp1/qwen_moe/data/gen_data.json", split="train",
                      field="instances") 

cnt = []
for sample in ds:
    for response in sample['responses']:
        embed = tokenizer.encode(response)
        cnt.append(len(embed))

print(np.sum(cnt))


