
import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, AutoModelForCausalLM
from accelerate import Accelerator

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
        default="/home/cyeab/ultra_onpolicy/Online-RLHF/alpaca_k8_sft_gen.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="alpaca_k8_sft_gen_pm.json",
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
device = accelerator.device

model = AutoModelForCausalLM.from_pretrained(script_args.reward_name_or_path,
                                             torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained("raftrsf/pair_pref", use_fast=True)


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
ds = load_dataset(data_files=ds_dir, split="train")

local_rank = Accelerator().local_process_index

data_size = len(ds["messages"])

share = int(data_size / world_size) + 1
ds = ds.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(ds))))
########
"""
We process the data format here and query the reward model to get the rewards.
"""
temperature=1.0


def comp(context, responses):
    probs_chosen = []
    for chosen_position in [0, 1]:
        # swap order to mitigate position bias
        response_A = responses[chosen_position]
        response_B = responses[1 - chosen_position]
        prompt = prompt_template.format(context=context, response_A=response_A, response_B=response_B)
        message = [
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model(input_ids)
        logit_A = output.logits[0, -1, token_id_A].item()
        logit_B = output.logits[0, -1, token_id_B].item()
        # take softmax to get the probability; using numpy
        Z = np.exp(logit_A / temperature) + np.exp(logit_B / temperature)
        logit_chosen = [logit_A, logit_B][chosen_position]
        prob_chosen = np.exp(logit_chosen / temperature) / Z
        probs_chosen.append(prob_chosen)
    avg_prob_chosen = np.mean(probs_chosen)
    
    if avg_prob_chosen == 0.5:
        #print("The ranking model gives equal probability for the following responses: ", responses)
        print("You")
    if avg_prob_chosen > 0.5:
        return 0
    elif avg_prob_chosen == 0.5:
        return 0.5
    else:
        return 1
    

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

def full_pairwise_ranking(prom, test_texts):
    # Initialize score dictionary
    scores = {txt: 0 for txt in test_texts}
    
    # Compare every pair of items
    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            result = comp(prom, [test_texts[i], test_texts[j]])
            if result == 0:
                scores[test_texts[i]] += 1
            elif result == 0.5:
                scores[test_texts[i]] += 0.5
                scores[test_texts[j]] += 0.5
            else:
                scores[test_texts[j]] += 1

    
    # Rank items based on scores in descending order
    #ranked_items = sorted(scores, key=scores.get, reverse=True)
    return scores





data = []


# tqdm is used to show the progress bar
with torch.no_grad():
    for sample in tqdm(ds):

        #tmp_prompt = sample['prompt']

        #test_texts = sample['responses']


        #map_original_texts = {test_texts[j]: [sample['responses'][j], sample['rewards'][j]] for j in range(len(test_texts))}
        #map_original_texts = {test_texts[j]: sample['responses'][j] for j in range(len(test_texts))}
        
        #scores = full_pairwise_ranking(tmp_prompt, test_texts)

        txt = sample['messages'][0]['content']

        #rewards = [scores[txt] for txt in test_texts]
        prob_A = get_prob_A(txt)

        data.append({"messages": sample['messages'], "prob_A": prob_A})


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

    output_eval_dataset = {}
    output_eval_dataset["type"] = "text_only"
    output_eval_dataset["instances"] = gathered_data
    with open(script_args.output_dir, "w", encoding="utf8") as f:
        json.dump(output_eval_dataset, f, ensure_ascii=False)

