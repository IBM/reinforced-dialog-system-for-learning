#!/bin/sh
mkdir -p trained_models/wiz-1.1/
BASE_PATH=../Talk_/

CUDA_VISIBLE_DEVICES=1 python doha.py \
  --data_dir ${BASE_PATH}data/WoW-processed/wiz/ \
  --experiment_type 'chat_document' \
  --do_train \
  --output_dir ${BASE_PATH}saved_models/wiz-1.1/ \
  --log_file_path ${BASE_PATH}logs/log_train_wiz_1.1.txt \
  --source_max_len 1024 \
  --target_max_len 128 \
  --train_batch_size 8 \
  --num_train_epochs 3 \
  --learning_rate 2e-5







