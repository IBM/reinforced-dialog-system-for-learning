mkdir trained_models/app-1.1/

CUDA_VISIBLE_DEVICES=0 python bart.py \
  --data_dir data/WoW-1.1/app/ \
  --experiment_type 'chat_document' \
  --do_train \
  --output_dir trained_models/app-1.1/ \
  --log_file_path trained_models/app-1.1/log.txt \
  --source_max_len 1024 \
  --target_max_len 128 \
  --train_batch_size 8 \
  --num_train_epochs 3 \

