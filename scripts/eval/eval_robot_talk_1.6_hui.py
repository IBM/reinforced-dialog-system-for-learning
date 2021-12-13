from datasets import load_dataset
from accelerate import Accelerator
import pickle
from utils.self_play_train_utils import *
from consts import *
from utils.eval_utils import eval_objective

# Prepare data and device
NUM_TEST = 1000
data_dir = '../Talk_/data/WoW-selector-1.5/'
data_files_rl = {
    "test": data_dir + 'test.json'
}
raw_datasets_rl = load_dataset('json', data_files=data_files_rl, field='data')
test_dataset = raw_datasets_rl['test'].select(range(NUM_TEST))
accelerator = Accelerator()
device = accelerator.device

scorer_cov = CoverageScorer()
alphas = [0.8, 0.2]
with open(BASE_PATH + 'za/args/args_coh.pkl', 'rb') as f:
    args_coh = pickle.load(f)
    args_coh.model_name_or_path = '../Talk_/saved_models/coh-1.0/pytorch_model.bin'
    scorer_coh = CoherenceScorer(args_coh, accelerator.device)

scorers = [scorer_cov, scorer_coh]

dump_dir = '../Talk_/saved_models/wiz-1.6.9/' # the saved model is not used
with open(dump_dir + 'args.pkl', 'rb') as f:
    args = pickle.load(f)

# apprentice
with open(BASE_PATH + 'za/args/args_bart_train.pkl', 'rb') as f:
    args_app = pickle.load(f)
    args_app.experiment_type = 'chat_document'
    args_app.model_file_path = args.app_path
    args_app.model_name = args.app_model_name
    app = BartQA(args_app, device)

# B. Cov only
dump_dir = '../Talk_/saved_models/wiz-1.6.9.1/'
result_dir = '../Talk_/results/1.6.9.1.txt'

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

# C. Coh only
dump_dir = '../Talk_/saved_models/wiz-1.7.5.2/'
result_dir = '../Talk_/results/1.7.coh.txt'


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
    wiz.generator.load_state_dict(torch.load(dump_dir + 'step_7200/pytorch_model.bin'))

# apprentice (Do not need to reaload)
trainer = RLTrainerForGenerator(args, wiz, app, scorers, alphas, accelerator)
test_dataloader = DataloaderRL(test_dataset, args.batch_size_rl, shuffle=False)
trainer.evaluate(test_dataloader)
