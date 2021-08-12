CUDA_VISIBLE_DEVICES=0 python self_play_apply.py \
  --data_dir data/WoW/wiz/ \
  --output_dir trained_models/wiz/doha/ \
  --model_path_app trained_models/wiz/doha/checkpoint-21000/model.pt \
  --model_path_wiz trained_models/wiz/doha/checkpoint-21000/model.pt \
  --model_path_sel trained_models/wiz/doha/checkpoint-21000/model.pt \
  --source_max_len 1024 \
  --target_max_len 128
