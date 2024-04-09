# vllm_eval

Set up the environment

```sh
conda create -n fix_vllm python=3.10.9
conda activate fix_vllm
pip3 install torch==2.1.2 torchvision torchaudio
git clone https://github.com/huggingface/alignment-handbook.git
cd alignment-handbook
git checkout d17fd7cd3b71c6a7bf7af34d8dc73135bb7ea8e9
python -m pip install .
python -m pip install flash-attn==2.3.6 --no-build-isolation
pip install vllm
pip install git+https://github.com/huggingface/transformers
```

## 1 Data generation

- Modify the base_dir and sft_model in run_scripts.sh
- The sampling parameters are set in line 68 of vllm_gen.py
- run the script

## 2 Compute tokens

- Modify the dataset dir in compute_token.py
  
```sh
python compute_token.py
```
