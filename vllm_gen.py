#!/usr/bin/env python
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk, load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, HfArgumentParser, pipeline, DataCollatorForSeq2Seq
from dataclasses import dataclass, field

from typing import Optional, List

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    model_name_or_path: Optional[str] = field(
        default="/home/xiongwei/gshf_gold_test/LMFlow_RAFT_Dev/output_models/online_dpo/new_iter2/checkpoint-pymodel2120",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="/home/xiongwei/gshf_gold_test/LMFlow_RAFT_Dev/output_models/online_dpo/iter1",
        metadata={"help": "the location of the dataset name or path"},
    )
    share: Optional[int] = field(
        default=999,
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    my_world_size: Optional[int] = field(
        default=4,
        metadata={"help": "the location of the dataset name or path"},
    )




parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model_path = script_args.model_name_or_path
print('model_path', model_path)
seed = 42
# set seed
torch.manual_seed(seed)
np.random.seed(seed)

llm = LLM(model=model_path,
          tokenizer=model_path,
          dtype='bfloat16', max_model_len=4096,
          #download_dir='/data/huggingface/',
          load_format='auto',
          seed=42
          # gpu_memory_utilization=0.95,
          )
tokenizer = AutoTokenizer.from_pretrained(model_path)

sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=2048, n=1, stop_token_ids=[tokenizer.eos_token_id], stop=["<|im_end|>", "<|im_start|>"]) 
ds = load_dataset("weqweasdas/ultra_prompt_split", split="prompt" + script_args.dataset_name_or_path).select(range(5000))

data_size = len(ds['prompt'])
one_num_share = int(data_size / script_args.my_world_size)
ds = ds.select(np.arange(script_args.share * one_num_share, (script_args.share + 1)*one_num_share))

print([script_args.share * one_num_share, (script_args.share + 1)*one_num_share])

print(ds, script_args.dataset_name_or_path)

# use tokenizer.apply_template to apply the template to the prompt
ds = ds.map(lambda x: {"final_prompt": tokenizer.apply_chat_template(
    x['prompt'], tokenize=False, add_generation_prompt=True)})


prompts = ds['final_prompt']
old_prompts = ds['prompt']


outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
import json

completions = []
used_prompts = []
gathered_data = []
for i, output in enumerate(outputs):
    tmp_data = {'old_prompt':old_prompts[i], "prompt": prompts[i], "responses": [out.text for out in output.outputs]}
    gathered_data.append(tmp_data)


output_eval_dataset = {}
output_eval_dataset['type'] = 'text_only'
output_eval_dataset['instances'] = gathered_data
print("I collect ", len(gathered_data), "samples")

with open(script_args.output_dir + "/gen_data" + str(script_args.share) + ".json", 'w', encoding='utf8') as f:
    json.dump(output_eval_dataset, f, ensure_ascii=False)


