
cd ..
export WANDB_PROJECT="GCC-caption"
export TOKENIZERS_PARALLELISM=false
for lora_targets in "attn,ffn"; do
    for decoder_type in base; do
        for lr in 5e-4; do
            deepspeed --include localhost:4 --master_port 55014 train.py \
            --output_dir "/outputs/${decoder_type}-${lora_targets}lora-lr${lr}-epoch4" \
            --do_train \
            --do_eval \
            --overwrite_output_dir 1 \
            --per_device_train_batch_size 128 \
            --per_device_eval_batch_size 128 \
            --gradient_accumulation_steps 1 \
            --remove_unused_columns 0 \
            --learning_rate ${lr} \
            --evaluation_strategy "steps" \
            --eval_steps 300 \
            --logging_steps 1 \
            --warmup_ratio 0.2 \
            --report_to "wandb" \
            --max_grad_norm 1.0 \
            --bf16 1 \
            --deepspeed "configs/ds_config.json" \
            --run_name "${decoder_type}-${lora_targets}lora-lr${lr}-epoch4" \
            --save_strategy "epoch" \
            --save_total_limit 1 \
            --num_train_epochs 4 \
            --lora_targets ${lora_targets} \
            --base_decoder_type "google/flan-t5-${decoder_type}"
        done
    done
done