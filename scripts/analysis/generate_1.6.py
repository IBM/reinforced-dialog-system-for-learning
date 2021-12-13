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
test_goals = {'relv_wow'}

scorer_cov = CoverageScorer()
alphas = [0.8, 0.2]
with open(BASE_PATH + 'za/args/args_coh.pkl', 'rb') as f:
    args_coh = pickle.load(f)
    args_coh.model_name_or_path = '../Talk_/saved_models/coh-1.0/pytorch_model.bin'
    scorer_coh = CoherenceScorer(args_coh, accelerator.device)

scorers = [scorer_cov, scorer_coh]

dump_dir = '../Talk_/saved_models/wiz-1.6.9.0/' # the saved model is not used
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
result_dir = '../Talk_/results/1.6.9.naive.txt'
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
eval_objective(data_dir + 'test.json', result_dir, NUM_TEST, device, {'read', 'rouge', 'qa', 'relv_unlearned', 'relv_wow'})
# Avg score of relv_unlearned: 0.5822140220999718

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
    wiz.generator.load_state_dict(torch.load(dump_dir + 'step_7200/pytorch_model.bin'))

# apprentice (Do not need to reaload)
trainer = RLTrainerForGenerator(args, wiz, app, scorers, alphas, accelerator)
test_dataloader = DataloaderRL(test_dataset, args.batch_size_rl, shuffle=False)
trainer.evaluate(test_dataloader)

eval_objective(data_dir + 'test.json', result_dir, NUM_TEST, device, test_goals)
# Avg score of relv_unlearned: 0.5750270951386602
# Avg score of relv_wow: 0.3076781860103698

# C. Coh only

dump_dir = '../Talk_/saved_models/wiz-1.6.9.2/'
result_dir = '../Talk_/results/1.6.9.2.txt'


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


# D. 1.6.9.5
dump_dir = '../Talk_/saved_models/wiz-1.6.9.5/'
result_dir = '../Talk_/results/1.6.9.5.txt'


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
# Avg score of relv_unlearned: 0.5759539510831236
# Avg score of relv_wow: 0.42185323682648596

# E. 1.6.9.3
dump_dir = '../Talk_/saved_models/wiz-1.6.9.3/'
result_dir = '../Talk_/results/1.6.9.3.txt'


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

eval_objective('../Talk_/data/WoW-selector-1.5/test.json', result_dir, NUM_TEST)


# scp glove_embedding.zip pengshancai@172.16.34.1:/home/pengshancai/.matchzoo/datasets/glove/

# F. 1.6.9.0
dump_dir = '../Talk_/saved_models/wiz-1.6.9.0/'
result_dir = '../Talk_/results/1.6.9.0.txt'
test_goals = {'read', 'rouge', 'qa', 'relv_unlearned'}

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

eval_objective('../Talk_/data/WoW-selector-1.5/test.json', result_dir, NUM_TEST, device, test_goals)



