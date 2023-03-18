#!/bin/bash

torchrun --nproc_per_node=4 --master_port=9292 train.py \
    --model_name_or_path /src/weights/llama-7b \
    --tokenizer_name_or_path /home/ubuntu/weights/llama-13b \
    --data_path /home/ubuntu/training-data \
    --bf16 True \
    --output_dir ~/alpaca_out \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
