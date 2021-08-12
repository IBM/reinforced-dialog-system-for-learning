CUDA_VISIBLE_DEVICES=0 python self_play_pretrain.py \
  --model_name_or_path 'roberta-base' \
  --train_file ../Talk_/data/WoW-selector-1.6/train.json \
  --validation_file ../Talk_/data/WoW-selector-1.6/dev.json \
  --log_file ../Talk_/logs/WoW-selector-self_play_pretrain-1.6.txt \
  --max_length 512 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --per_device_eval_batch_size=2 \
  --per_device_train_batch_size=2 \
  --output_dir ../Talk_/saved_models/sel-1.6

