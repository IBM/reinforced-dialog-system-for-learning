from datasets import load_dataset
from accelerate import Accelerator
import pickle
from utils.self_play_train_utils import *
from consts import *
from utils.eval_utils import eval_objective

# Prepare data and device
NUM_TEST = 1000
data_dir = '../Talk_/data/CNN-DailyMail-processed/'
data_files_rl = {
    "test": data_dir + 'test.json'
}
raw_datasets_rl = load_dataset('json', data_files=data_files_rl, field='data')
test_dataset = raw_datasets_rl['test'].select(range(NUM_TEST))
accelerator = Accelerator()
device = accelerator.device
test_goals = {'relv_unlearned', 'relv_wow'}

scorer_cov = CoverageScorer()
alphas = [0.8, 0.2]
with open(BASE_PATH + 'za/args/args_coh.pkl', 'rb') as f:
    args_coh = pickle.load(f)
    args_coh.model_name_or_path = '../Talk_/saved_models/coh-1.0/pytorch_model.bin'
    scorer_coh = CoherenceScorer(args_coh, accelerator.device)

scorers = [scorer_cov, scorer_coh]

dump_dir = '../Talk_/saved_models/wiz-1.7.6/' # the saved model is not used
with open(dump_dir + 'args.pkl', 'rb') as f:
    args = pickle.load(f)


# apprentice
with open(BASE_PATH + 'za/args/args_bart_train.pkl', 'rb') as f:
    args_app = pickle.load(f)
    args_app.experiment_type = 'chat_document'
    args_app.model_file_path = args.app_path
    args_app.model_name = args.app_model_name
    app = BartQA(args_app, device)


# A. Naive
result_dir = '../Talk_/results/1.7.5.naive.txt'
args.log_dir = result_dir

# wizard
with open(BASE_PATH + 'za/args/args_doha_train.pkl', 'rb') as f:
    args_wiz = pickle.load(f)
    args_wiz.learning_rate = args.learning_rate
    args_wiz.experiment_type = 'chat_document'
    args_wiz.model_file_path = args.wiz_path
    args_wiz.model_name = args.wiz_model_name
    wiz = MultiBartQA(args_wiz, device)
# wiz.generator.load_state_dict(torch.load(dump_dir + 'step_9900/pytorch_model.bin'))

trainer = RLTrainerForGenerator(args, wiz, app, scorers, alphas, accelerator)
test_dataloader = DataloaderRL(test_dataset, args.batch_size_rl, shuffle=False)
trainer.evaluate(test_dataloader)
eval_objective(data_dir + '/test.json', result_dir, NUM_TEST, device, test_goals=test_goals)
# Avg score of relv_unlearned: 0.6055110518485308
# Avg score of relv_wow: 0.5217163836632972


# B. Cov only
dump_dir = '../Talk_/saved_models/wiz-1.7.6.1/'
result_dir = '../Talk_/results/1.7.6.1.txt'

with open(dump_dir + 'args.pkl', 'rb') as f:
    args = pickle.load(f)
    args.log_dir = result_dir

with open(BASE_PATH + 'za/args/args_doha_train.pkl', 'rb') as f:
    args_wiz = pickle.load(f)
    args_wiz.learning_rate = args.learning_rate
    args_wiz.experiment_type = 'chat_document'
    args_wiz.model_file_path = args.wiz_path
    args_wiz.model_name = args.wiz_model_name
    wiz = MultiBartQA(args_wiz, device)
    wiz.generator.load_state_dict(torch.load(dump_dir + 'step_9900/pytorch_model.bin'))

# apprentice (Do not need to reaload)
trainer = RLTrainerForGenerator(args, wiz, app, scorers, alphas, accelerator)
test_dataloader = DataloaderRL(test_dataset, args.batch_size_rl, shuffle=False)
trainer.evaluate(test_dataloader)
eval_objective(data_dir + '/test.json', result_dir, NUM_TEST, device, test_goals=test_goals)
# Avg score of relv_unlearned: 0.5987194020301103
# Avg score of relv_wow: 0.2539130405058968


# C. Coh only
dump_dir = '../Talk_/saved_models/wiz-1.7.6.2/'
result_dir = '../Talk_/results/1.7.6.2.txt'


with open(dump_dir + 'args.pkl', 'rb') as f:
    args = pickle.load(f)
    args.log_dir = result_dir

# wizard
with open(BASE_PATH + 'za/args/args_doha_train.pkl', 'rb') as f:
    args_wiz = pickle.load(f)
    args_wiz.learning_rate = args.learning_rate
    args_wiz.experiment_type = 'chat_document'
    args_wiz.model_file_path = args.wiz_path
    args_wiz.model_name = args.wiz_model_name
    wiz = MultiBartQA(args_wiz, device)
    wiz.generator.load_state_dict(torch.load(dump_dir + 'step_9900/pytorch_model.bin'))

# apprentice (Do not need to reaload)
trainer = RLTrainerForGenerator(args, wiz, app, scorers, alphas, accelerator)
test_dataloader = DataloaderRL(test_dataset, args.batch_size_rl, shuffle=False)
trainer.evaluate(test_dataloader)
eval_objective(data_dir + '/test.json', result_dir, NUM_TEST)


# D. Full
dump_dir = '../Talk_/saved_models/wiz-1.7.6.3/'
result_dir = '../Talk_/results/1.7.6.3.txt'


with open(dump_dir + 'args.pkl', 'rb') as f:
    args = pickle.load(f)
    args.log_dir = result_dir

# wizard
with open(BASE_PATH + 'za/args/args_doha_train.pkl', 'rb') as f:
    args_wiz = pickle.load(f)
    args_wiz.learning_rate = args.learning_rate
    args_wiz.experiment_type = 'chat_document'
    args_wiz.model_file_path = args.wiz_path
    args_wiz.model_name = args.wiz_model_name
    wiz = MultiBartQA(args_wiz, device)
    wiz.generator.load_state_dict(torch.load(dump_dir + 'step_9900/pytorch_model.bin'))

# apprentice (Do not need to reaload)
trainer = RLTrainerForGenerator(args, wiz, app, scorers, alphas, accelerator)
test_dataloader = DataloaderRL(test_dataset, args.batch_size_rl, shuffle=False)
trainer.evaluate(test_dataloader)
eval_objective(data_dir + '/test.json', result_dir, NUM_TEST, device, test_goals=test_goals)
# Avg score of relv_unlearned: 0.609204359561623
# Avg score of relv_wow: 0.3151311956450785

# Avg score of rouge1-0_gain: 0.3255732488944512
# Avg score of rouge2-0_gain: 0.3070189980473422
# Avg score of rougeL-0_gain: 0.3188341653168811
# Avg score of rouge1-1_gain: 0.23932318890716475
# Avg score of rouge2-1_gain: 0.22901753131702515
# Avg score of rougeL-1_gain: 0.21993498988437157
# Avg score of rouge1-2_gain: 0.15268613398974706
# Avg score of rouge2-2_gain: 0.14550904763737307
# Avg score of rougeL-2_gain: 0.12094929282389877
# Avg score of rouge1: 0.7175825717913631
# Avg score of rouge2: 0.6815455770017405
# Avg score of rougeL: 0.6597184480251514
# Avg score of cov: 0.7209780000000001
# Avg score of coh: 0.40483299999999994
# Avg score of avg_prob: 0.2883214727664768
# Avg score of avg_fscore: 0.36177510709820376
# Avg score of len-0: 28.7
# Avg score of len-1: 31.416
# Avg score of len-2: 26.381
# Avg score of len-2: 86.497 / 28.83


