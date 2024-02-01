#!/bin/bash
cd ..
export CUDA_VISIBLE_DEVICES=5
for model_name in "base-attn,ffnlora-lr5e-4-epoch4" "base-attnlora-lr5e-4-epoch4" "large-attlora-lr5e-4-epoch4" "large-attn,ffnlora-lr5e-4-epoch4"; do
    python best_of_n_sample.py --checkpoint /outputs/${model_name}/checkpoint-5088
done