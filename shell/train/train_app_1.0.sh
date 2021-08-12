CUDA_VISIBLE_DEVICES=0,1 python bart.py \
  --data_dir data/WoW-1.0/app/ \
  --experiment_type 'chat_document' \
  --do_train \
  --output_dir trained_models/app-1.0/ \
  --log_file_path trained_models/app-1.0/log.txt \
  --source_max_len 1024 \
  --target_max_len 128 \
  --train_batch_size 8 \
  --num_train_epochs 3 \

