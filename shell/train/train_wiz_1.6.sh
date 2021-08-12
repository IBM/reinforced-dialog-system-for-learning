mkdir -p ../Talk_/saved_models/wiz-1.6/

CUDA_VISIBLE_DEVICES=1 python doha.py \
  --data_dir ../Talk_/data/WoW-1.6/wiz/ \
  --experiment_type 'chat_document' \
  --do_train \
  --output_dir ../Talk_/saved_models/wiz-1.6/ \
  --log_file_path ../Talk_/saved_models/wiz-1.6/log.txt \
  --source_max_len 1024 \
  --target_max_len 128 \
  --train_batch_size 8 \
  --num_train_epochs 3 \
  --learning_rate 2e-5



