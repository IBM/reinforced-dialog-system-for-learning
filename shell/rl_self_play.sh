python self_play_rl_generator.py \
--train_file_rl path_to_train_set_of_Wikipedia/CnnDailyMail/PaperAbstracts \
# e.g. --train_file_rl ../Talk_/data/Papers-processed/train.json \
--validation_file_rl path_to_train_set_of_Wikipedia/CnnDailyMail/PaperAbstracts \
# e.g. --validation_file_rl ../Talk_/data/Papers-processed/dev.json \
--cached_train_file_mle path_to_cached_file_for_optimizing_MLE_loss \
# e.g. --cached_train_file_mle ../Talk_/data/WoW-processed/wiz/cached_train \
--output_dir ../Talk_/saved_models/TeacherBot/ \
--log_dir ../Talk_/logs/self_play_rl.txt \
--max_train_steps=10000 \
--save_steps 900 \
--eval_steps 300 \
--warmup_steps 600 \
--batch_size_rl 5 \
--batch_size_mle 8 \
--max_cov_score 0.3 \
--num_cached_responses 5 \
--learning_rate 1e-6 \
--wiz_path path_to_teacher_bot_pre-trained_on_WoW_dataset \
# e.g. --wiz_path ../Talk_/saved_models/TeacherBotPreTuned/model.pt \
--app_path path_to_student_bot_pre-trained_on_WoW_dataset \
# e.g. --app_path ../Talk_/saved_models/StudentBotPreTuned/model.pt \
--coh_model wow \
--coh_path path_to_coherence_evaluation_model_pre-trained_on_WoW__coherence_dataset \
# e.g. --coh_path ../Talk_/saved_models/CoherenceModelPretrained/pytorch_model.bin \
--num_turns 3 \
--eval_size 50 \
--alpha_cov 0.7 \
--alpha_coh 0.3 \
--num_candicates 10 \
--num_mle_per_rl 3

