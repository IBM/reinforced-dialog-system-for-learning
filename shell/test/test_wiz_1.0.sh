CUDA_VISIBLE_DEVICES=0 python doha.py \
  --data_dir data/WoW/wiz/ \
  --experiment_type 'chat_document' \
  --do_generate \
  --output_dir trained_models/wiz/doha/ \
  --model_file_path trained_models/wiz/doha/checkpoint-21000/model.pt \
  --source_max_len 1024 \
  --target_max_len 128
