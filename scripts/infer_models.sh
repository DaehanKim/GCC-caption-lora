#!/bin/bash
cd ..
export CUDA_VISIBLE_DEVICES=6
for ckpt in "/outputs/base-attn,ffnlora-lr5e-4-epoch4/checkpoint-5088" "/outputs/base-attnlora-lr5e-4-epoch4/checkpoint-5088"; do
    for decoding in "beam" "sample"; do
        python evaluation.py --checkpoint ${ckpt} --decoding ${decoding}
    done
done
