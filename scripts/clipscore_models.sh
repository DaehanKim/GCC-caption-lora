#!/bin/bash
cd ..
export CUDA_VISIBLE_DEVICES=7
for model_name in "base-attn,ffnlora-lr5e-4-epoch4" "base-attnlora-lr5e-4-epoch4" "large-attlora-lr5e-4-epoch4" "large-attn,ffnlora-lr5e-4-epoch4"; do
    for decoding in "bestof10"; do
        python clipscore.py --hyp_text inference_results/${model_name}_${decoding}_hyp.txt
    done
done
