python self_play_pretrain.py \
  --model_name_or_path prajjwal1/bert-tiny \
  --train_file data/WoW-selector-1.1/train.json \
  --validation_file data/WoW-selector-1.1/dev.json \
  --max_length 512 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --per_device_eval_batch_size=8 \
  --per_device_train_batch_size=8 \
  --output_dir saved_models/sel-1.1 \
  --num_neg_per_pos 11

