CUDA_VISIBLE_DEVICES=2 python self_play_rl.py \
--train_file ../Talk_/data/WoW-selector-1.5/train.json \
--validation_file ../Talk_/data/WoW-selector-1.5/dev.json \
--learning_rate 5e-5 \
--output_dir ../Talk_/saved_models/sel-1.5.3 \
--log_dir ../Talk_/logs/self_play_rl_1.5.3.txt \
--max_train_steps=100000 \
--save_steps 600 \
--eval_steps 600 \
--sel_path roberta-base \
--wiz_path ../Talk_/saved_models/wiz-1.1/checkpoint-26000/model.pt \
--app_path ../Talk_/saved_models/app-1.1/checkpoint-23000/model.pt \
--coh_path ../Talk_/saved_models/coh-1.0/pytorch_model.bin \
--num_turns 3 \
--eval_size 50 \
--alpha_cov 0.7 \
--alpha_coh 0.3 \
--num_candicates 16 \
--selector_type post

