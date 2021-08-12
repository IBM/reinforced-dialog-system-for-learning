python self_play_rl.py \
--model_name_or_path roberta-base \
--train_file ../Talk_/data/WoW-selector-1.5/train.json \
--validation_file ../Talk_/data/WoW-selector-1.5/dev.json \
--learning_rate 5e-5 \
--output_dir ../Talk_/saved_models/sel-1.5 \
--log_dir ../Talk_/logs/logs_1.5.txt \
--max_train_steps=100000 \
--save_steps 600 \
--eval_steps 600 \
--wiz_path saved_models/wiz-1.1/checkpoint-26000/model.pt \
--app_path saved_models/app-1.1/checkpoint-23000/model.pt \
--num_turns 2 \
--eval_size 50 \
--num_candicates 16


