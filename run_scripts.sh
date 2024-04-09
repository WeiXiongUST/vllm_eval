#!/bin/bash
#conda init bash
base_dir="/home/wx/exp1"
sft_model="Qwen/Qwen1.5-MoE-A2.7B"

mkdir $base_dir

i=0
next_i=$((i + 1))
model_dir=${base_dir}/model${i}
mkdir ${model_dir}
mkdir ${model_dir}/data

my_world_size=4
CUDA_VISIBLE_DEVICES=0 python vllm_gen.py --model_name_or_path ${model_dir} --dataset_name_or_path ${i} --output_dir ${model_dir}/data --share 0 --my_world_size ${my_world_size} &
CUDA_VISIBLE_DEVICES=1 python vllm_gen.py --model_name_or_path ${model_dir} --dataset_name_or_path ${i} --output_dir ${model_dir}/data --share 1 --my_world_size ${my_world_size} &
CUDA_VISIBLE_DEVICES=2 python vllm_gen.py --model_name_or_path ${model_dir} --dataset_name_or_path ${i} --output_dir ${model_dir}/data --share 2 --my_world_size ${my_world_size} &
CUDA_VISIBLE_DEVICES=3 python vllm_gen.py --model_name_or_path ${model_dir} --dataset_name_or_path ${i} --output_dir ${model_dir}/data --share 3 --my_world_size ${my_world_size} &


wait 


python ./new_transform.py --dataset_path ${model_dir}/data/gen_data --my_idx ${i} --output_dir ${model_dir}/gen_data
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch ./get_reward.py --dataset_name_or_path ${model_dir}/data/gen_data.json --output_dir ${model_dir}/data --record_dir ${base_dir}/reward_record.txt 

