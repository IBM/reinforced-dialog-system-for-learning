from datasets import load_dataset
from accelerate import Accelerator
import pickle
from utils.self_play_train_utils import *
from consts import *
from utils.eval_utils import eval_objective

accelerator = Accelerator()
device = accelerator.device

# A. Naive
dump_dir = '../Talk_/saved_models/wiz-1.8.3/' # the saved model is not used
data_dir = '../Talk_/data/Papers-processed/'
result_dir = '../Talk_/results/1.8.3.naive.txt'
NUM_TEST = 1000
test_goals = {'relv_wow'}


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
# wiz.generator.load_state_dict(torch.load(dump_dir + 'step_9900/pytorch_model.bin'))

# apprentice
with open(BASE_PATH + 'za/args/args_bart_train.pkl', 'rb') as f:
    args_app = pickle.load(f)
    args_app.experiment_type = 'chat_document'
    args_app.model_file_path = args.app_path
    args_app.model_name = args.app_model_name
    app = BartQA(args_app, device)

scorer_cov = CoverageScorer()
with open(BASE_PATH + 'za/args/args_coh.pkl', 'rb') as f:
    args_coh = pickle.load(f)
    args_coh.model_name_or_path = args.coh_path
    scorer_coh = CoherenceScorer(args_coh, accelerator.device)

scorers = [scorer_cov, scorer_coh]
alphas = [args.alpha_cov, args.alpha_coh]
trainer = RLTrainerForGenerator(args, wiz, app, scorers, alphas, accelerator)

data_files_rl = {
    "test": data_dir + 'test.json'
}
raw_datasets_rl = load_dataset('json', data_files=data_files_rl, field='data')
test_dataset = raw_datasets_rl['test']
test_dataloader = DataloaderRL(test_dataset, args.batch_size_rl, shuffle=False)
trainer.evaluate(test_dataloader)
eval_objective(data_dir + 'test.json', result_dir, NUM_TEST, device, test_goals)
# Avg score of relv_unlearned: 0.6012862303853035
# Avg score of relv_wow: 0.5655843334191013


# B. Cov only
dump_dir = '../Talk_/saved_models/wiz-1.8.3.1/'
result_dir = '../Talk_/results/1.8.3.cov.txt'


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
eval_objective(data_dir + 'test.json', result_dir, NUM_TEST, device, test_goals)
# Avg score of relv_unlearned: 0.589322346240282
# Avg score of relv_wow: 0.27202351595601065

# C. Coh only
dump_dir = '../Talk_/saved_models/wiz-1.8.3.2/'
result_dir = '../Talk_/results/1.8.3.coh.txt'


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
eval_objective(data_dir + 'test.json', result_dir, NUM_TEST, device, test_goals)
# Avg score of relv_unlearned: 0.6296369413435459
# Avg score of relv_wow: 0.7378453355757519

# D. Full
dump_dir = '../Talk_/saved_models/wiz-1.8.3/'
result_dir = '../Talk_/results/1.8.3.full.txt'


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
eval_objective(data_dir + 'test.json', result_dir, NUM_TEST, device, test_goals)
# Avg score of relv_unlearned: 0.6093919047514998
# Avg score of relv_wow: 0.407213192785416
