mkdir trained_models/app-1.1/
BASE_PATH=../Talk_/

CUDA_VISIBLE_DEVICES=3 python bart.py \
  --data_dir ${BASE_PATH}data/WoW-processed/app/ \
  --experiment_type 'chat_document' \
  --do_train \
  --output_dir ${BASE_PATH}saved_models/app-1.1/ \
  --log_file_path ${BASE_PATH}logs/log_train_app_1.1.txt \
  --source_max_len 1024 \
  --target_max_len 128 \
  --train_batch_size 8 \
  --num_train_epochs 3 \



