
import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, AutoModelForCausalLM
from accelerate import Accelerator
from datasets import load_dataset, Dataset

tqdm.pandas()

#####
# This script takes a dataset as the input, where each sample is {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...]}
# The script will compute the reward for each input-output pair, and eventually output a new dataset, where each sample contains {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...], "rewards": [reward1, reward2, ...]}
#####


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    dataset_name_or_path: Optional[str] = field(
        default="RLHFlow/pair_preference_model_dataset",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="weqweasdas/xxxx_test",
        metadata={"help": "the location of the output file"},
    )
    # No need to set record_dir
    record_dir: Optional[str] = field(
        default=None, 
        metadata={"help": "the location of the recording file"},
    )
    reward_name_or_path: Optional[str] = field(
        default="RLHFlow/pair-preference-model-LLaMA3-8B",
        metadata={"help": "the name of the reward model"},
    )
    input_output_delimiter: Optional[str] = field(
        default="",
        metadata={"help": "the delimiter between input and output"},
    )
    K: Optional[int] = field(
        default=1,
        metadata={"help": "the number of responses per prompt"},
    )


accelerator = Accelerator()

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

device = accelerator.device

model = AutoModelForCausalLM.from_pretrained(script_args.reward_name_or_path,
                                             torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(script_args.reward_name_or_path, use_fast=True)


prompt_template = "[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"
token_id_A = tokenizer.encode("A", add_special_tokens=False)
token_id_B = tokenizer.encode("B", add_special_tokens=False)
assert len(token_id_A) == 1 and len(token_id_B) == 1
token_id_A = token_id_A[0]
token_id_B = token_id_B[0]

ds_dir = script_args.dataset_name_or_path
world_size = int(os.getenv("WORLD_SIZE", "1"))

####### The dataset is supposed to contain {"prompt": prompt, "responses": [response1, response2, ..., responsek]}
# No additional format! Just raw prompt and response.
ds = load_dataset(ds_dir, split="train").select(range(100))

local_rank = Accelerator().local_process_index

data_size = len(ds["messages"])

share = int(data_size / world_size) + 1
ds = ds.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(ds))))
########
"""
We process the data format here and query the reward model to get the rewards.
"""
temperature=1.0

    

def get_prob_A(txt):
    message = [
        {"role": "user", "content": txt},
    ]
    input_ids = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model(input_ids)
    logit_A = output.logits[0, -1, token_id_A].item()
    logit_B = output.logits[0, -1, token_id_B].item()
    # take softmax to get the probability; using numpy
    Z = np.exp(logit_A / temperature) + np.exp(logit_B / temperature)
    logit_chosen = [logit_A, logit_B][0]
    prob_chosen = np.exp(logit_chosen / temperature) / Z
    return prob_chosen




data = []
z = 0
# tqdm is used to show the progress bar
with torch.no_grad():
    for sample in tqdm(ds):
        
        txt = sample['messages'][0]['content']

        prob_A = get_prob_A(txt)

        data.append({"messages": sample['messages'], "prob_A": prob_A})
        if z == 0:
            print("we print an example to monitor the process")
            tmp_message = [
                {"role": "user", "content": txt},
            ]
            print(tokenizer.apply_chat_template(tmp_message, add_generation_prompt=True, tokenize=False))
            z += 1
            
# Send the data to other GPUs
world_size = int(os.getenv("WORLD_SIZE", "1"))
all_process_list = [{}] * world_size

data_to_send = {
    "data": [[data[i]] for i in range(len(data))],
}

import torch.distributed as dist

dist.all_gather_object(all_process_list, data_to_send)
gathered_data = []


for i in range(world_size):
    tmp_data = [tmp[0] for tmp in all_process_list[i]["data"]]
    gathered_data.extend(tmp_data)

all_rewards = [sample["prob_A"] for sample in gathered_data]
mean_scores = np.mean(all_rewards)


if local_rank == 0:
    print(
        "Collect {} data from {} inputs. mean score {} ".format(
            len(gathered_data), data_size, mean_scores,
        )
    )

    dict_data = {
        "messages": [d['messages'] for d in gathered_data],
        "prob_A": [d['prob_A'] for d in gathered_data]
    }
    dataset = Dataset.from_dict(dict_data)
    DatasetDict({'train': dataset}).push_to_hub(script_args.output_dir)
    
    #output_eval_dataset = {}
    #output_eval_dataset["type"] = "text_only"
    #output_eval_dataset["instances"] = gathered_data
    #with open(script_args.output_dir, "w", encoding="utf8") as f:
    #    json.dump(output_eval_dataset, f, ensure_ascii=False)

