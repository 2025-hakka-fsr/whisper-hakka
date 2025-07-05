#!/bin/bash

torchrun --nproc_per_node=1 train/fine-tune_on_custom_dataset.py \
--model_name openai/whisper-tiny \
--language en \
--sampling_rate 16000 \
--num_proc 1 \
--train_strategy epoch \
--learning_rate 5e-4 \
--warmup 50 \
--train_batchsize 8 \
--eval_batchsize 4 \
--gradient_accumulation_steps 4 \
--num_epochs 1 \
--resume_from_ckpt None \
--output_dir outputs/whisper-hakka-tiny-lora \
--train_datasets formatted_data_pinyin/train_dataset_sm  \
--eval_datasets formatted_data_pinyin/eval_dataset_sm  \
--use_lora \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1
