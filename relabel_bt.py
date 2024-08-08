
import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
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
        default="princeton-nlp/gemma2-ultrafeedback-armorm",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="gemma2_it_gen_bt_rm_label.json",
        metadata={"help": "the location of the output file"},
    )
    # No need to set record_dir
    record_dir: Optional[str] = field(
        default=None, 
        metadata={"help": "the location of the recording file"},
    )
    reward_name_or_path: Optional[str] = field(
        default="sfairXC/FsfairX-LLaMA3-RM-v0.1",
        metadata={"help": "the name of the reward model"},
    )
    input_output_delimiter: Optional[str] = field(
        default="",
        metadata={"help": "the delimiter between input and output"},
    )
    K: Optional[int] = field(
        default=5,
        metadata={"help": "the number of responses per prompt"},
    )


accelerator = Accelerator()

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

device = accelerator.device
pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 1,
}
reward_model = script_args.reward_name_or_path
rm_tokenizer = AutoTokenizer.from_pretrained(reward_model)
rm_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model,
    device=device,
    tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    truncation=True,
)


ds_dir = script_args.dataset_name_or_path
world_size = int(os.getenv("WORLD_SIZE", "1"))

####### The dataset is supposed to contain {"prompt": prompt, "responses": [response1, response2, ..., responsek]}
# No additional format! Just raw prompt and response.
ds = load_dataset(ds_dir, split="test").select(range(200))

local_rank = Accelerator().local_process_index

data_size = len(ds["prompt"])

share = int(data_size / world_size) + 1

ds = ds.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(ds))))

########
"""
We process the data format here and query the reward model to get the rewards.
"""


def get_reward(test_texts):
    pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
    rewards = [output[0]["score"] for output in pipe_outputs]
    return rewards


def change_of_format(prom, resp):
    # To be modified according to the reward model and the LLM you use
    # Be careful about multi-turn conversions
    """
    prom = prom.replace("<s>GPT4 Correct User: ", "").replace("<|end_of_turn|>GPT4 Correct Assistant:", "")

    final_resp = resp.split("GPT4 Correct User")[0]
    """
    prom = prom
    final_resp = resp

    message = [
        {"role": "user", "content": prom},
        {"role": "assistant", "content": final_resp},
    ]
    return rm_tokenizer.apply_chat_template(message, tokenize=False).replace(rm_tokenizer.bos_token, "")


data = []

# tqdm is used to show the progress bar
with torch.no_grad():
    for sample in tqdm(ds):
        # The VLLM may not generate responses for some prompts because it is too long, we skip them
        if len(sample["all_generated_responses"]) < script_args.K:
            print("the num of responses is less than K")
            continue
        test_texts = [change_of_format(sample['prompt'], tmp_output) for tmp_output in sample['all_generated_responses']]
        rewards = get_reward(test_texts)
        data.append({"prompt": sample["prompt"], "responses": sample["all_generated_responses"], "rewards": rewards})


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


if local_rank == 0:
    output_eval_dataset = {}
    output_eval_dataset["type"] = "text_only"
    output_eval_dataset["instances"] = gathered_data
    with open(script_args.output_dir, "w", encoding="utf8") as f:
        json.dump(output_eval_dataset, f, ensure_ascii=False)

