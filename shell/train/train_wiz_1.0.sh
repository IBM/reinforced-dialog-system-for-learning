CUDA_VISIBLE_DEVICES=0 python doha.py \
  --data_dir data/WoW-1.0/wiz/ \
  --experiment_type 'chat_document' \
  --do_train \
  --output_dir trained_models/wiz-1.0/ \
  --log_file_path trained_models/wiz-1.0/log.txt \
  --source_max_len 1024 \
  --target_max_len 128 \
  --train_batch_size 8 \
  --num_train_epochs 3 \
  --learning_rate 2e-5



