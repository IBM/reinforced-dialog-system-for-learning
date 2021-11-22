CUDA_VISIBLE_DEVICES=3 python self_play_rl_generator.py \
--train_file_rl ../Talk_/data/WoW-selector-1.5/train.json \
--validation_file_rl ../Talk_/data/WoW-selector-1.5/dev.json \
--cached_train_file_mle ../Talk_/data/WoW-processed/WoW-1.1.1/wiz/cached_train \
--learning_rate 1e-5 \
--output_dir ../Talk_/saved_models/wiz-1.6.1/ \
--log_dir ../Talk_/logs/self_play_rl_1.6.1.txt \
--max_train_steps=50000 \
--save_steps 300 \
--eval_steps 300 \
--warmup_steps 600 \
--batch_size_rl 5 \
--batch_size_mle 8 \
--learning_rate 5e-5 \
--wiz_path ../Talk_/saved_models/wiz-1.1.1/checkpoint-23000/model.pt \
--app_path ../Talk_/saved_models/app-1.1.1/checkpoint-29000/model.pt \
--coh_path ../Talk_/saved_models/coh-1.0/pytorch_model.bin \
--num_turns 3 \
--eval_size 50 \
--alpha_cov 1.0 \
--alpha_coh 0.0 \
--num_candicates 10


