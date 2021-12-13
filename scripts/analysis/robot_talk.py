import json

import datasets
from datasets import load_dataset, load_metric
from transformers import (
    MODEL_MAPPING
)
from accelerate import Accelerator
import pickle
from utils.self_play_infra_utils import *
from utils.self_play_train_utils import *
from consts import *

accelerator = Accelerator()
device = accelerator.device
target_file = '../Talk_/za/examples/examples.txt'

'''
1.6
'''
version = 'wiz-1.6.9.5'
wiz_path = '../Talk_/saved_models/%s/step_9900/pytorch_model.bin' % version


with open(BASE_PATH + 'saved_models/%s/args.pkl' % version, 'rb') as f:
    args = pickle.load(f)

with open(BASE_PATH + 'za/args/args_doha_train.pkl', 'rb') as f:
    args_wiz = pickle.load(f)
    args_wiz.experiment_type = 'chat_document'
    args_wiz.model_file_path = args.wiz_path
    args_wiz.model_name = args.wiz_model_name
    wiz = MultiBartQA(args_wiz, device)
    if wiz_path != None:
        wiz.generator.load_state_dict(torch.load(wiz_path))

with open(BASE_PATH + 'za/args/args_bart_train.pkl', 'rb') as f:
    args_app = pickle.load(f)
    args_app.experiment_type = 'chat_document'
    args_app.model_file_path = '../Talk_/saved_models/app-1.1.1/checkpoint-29000/model.pt'
    args_app.model_name = args.app_model_name
    app = BartQA(args_app, device)

scorer_cov = CoverageScorer()
# coherence scorer
with open(BASE_PATH + 'za/args/args_coh.pkl', 'rb') as f:
    args_coh = pickle.load(f)
    args_coh.model_name_or_path = '../Talk_/saved_models/coh-1.0/pytorch_model.bin'

scorer_coh = CoherenceScorer(args_coh, accelerator.device)
scorers = [scorer_cov, scorer_coh]

wiz, app = accelerator.prepare(wiz, app)
alphas = [args.alpha_cov, args.alpha_coh]
trainer = RLTrainerForGenerator(args, wiz, app, scorers, alphas, accelerator)

topics = ['Lloyd Cole (album)',]
documents = ['Lloyd Cole, also known as The X Album, is the debut solo album by English singer, songwriter and musician Lloyd Cole released on February 21, 1990 by Polydor and Capitol. Previously Cole was best known for his work with The Commotions but this album marked a departure from their signature sound and an opportunity for him to collaborate with other musicians and explore new ideas. Following the release of "Mainstream", and limited touring and promotion in support of it, Cole left the Commotions. The chance to write and record new types of songs was the motivating factor. He also believed that the group had reached a natural conclusion and he no longer wanted the pressure and responsibility of managing a band.']
histories = ['']

for j in range(3):
    wiz_says = trainer.generate_wiz_response(topics, histories, documents)
    histories = trainer.update_histories(wiz_says, histories, reverse=args.reverse)
    if not j == trainer.args.num_turns - 1:
        app_says = trainer.generate_app_response(histories)
        histories = trainer.update_histories(app_says, histories, reverse=args.reverse)

uttrs = histories[0].split(' / ')
uttrs.reverse()

with open(target_file, 'a') as f:
    _ = f.write('Topic:\t%s\n' % topics[0])
    _ = f.write('Document:\t%s\nHistory:\n' % documents[0])
    for j, uttr in enumerate(uttrs):
        if j % 2 == 0:    # Wiz
            _ = f.write('\tTeacher - %s\n' % uttr)
        else:
            _ = f.write('\tStudent - %s\n' % uttr)
    _ = f.write('\n')



'''
1.7
'''
version = 'wiz-1.7.6.3'
wiz_path = '../Talk_/saved_models/%s/step_9900/pytorch_model.bin' % version


with open(BASE_PATH + 'saved_models/%s/args.pkl' % version, 'rb') as f:
    args = pickle.load(f)

with open(BASE_PATH + 'za/args/args_doha_train.pkl', 'rb') as f:
    args_wiz = pickle.load(f)
    args_wiz.experiment_type = 'chat_document'
    args_wiz.model_file_path = args.wiz_path
    args_wiz.model_name = args.wiz_model_name
    wiz = MultiBartQA(args_wiz, device)
    if wiz_path != None:
        wiz.generator.load_state_dict(torch.load(wiz_path))

with open(BASE_PATH + 'za/args/args_bart_train.pkl', 'rb') as f:
    args_app = pickle.load(f)
    args_app.experiment_type = 'chat_document'
    args_app.model_file_path = '../Talk_/saved_models/app-1.1.1/checkpoint-29000/model.pt'
    args_app.model_name = args.app_model_name
    app = BartQA(args_app, device)

scorer_cov = CoverageScorer()
# coherence scorer
with open(BASE_PATH + 'za/args/args_coh.pkl', 'rb') as f:
    args_coh = pickle.load(f)
    args_coh.model_name_or_path = '../Talk_/saved_models/coh-1.0/pytorch_model.bin'

scorer_coh = CoherenceScorer(args_coh, accelerator.device)
scorers = [scorer_cov, scorer_coh]

wiz, app = accelerator.prepare(wiz, app)
alphas = [args.alpha_cov, args.alpha_coh]
trainer = RLTrainerForGenerator(args, wiz, app, scorers, alphas, accelerator)

topics = ["Presidential hopeful's video, featuring gay couple, gets mature rating in Russia "]
documents = ['Moscow A Russian TV channel aired Hillary Clinton\'s first campaign video with a rating stamp that means it\'s for mature audiences, because of fears it might run afoul of the country\'s anti-gay propaganda law. A clip of the video, which features a gay couple holding hands, got the 18+ rating from the independent TV Rain channel in Russia on Monday. The channel told CNN that it didn\'t want to break the controversial law, which bans "propaganda of nontraditional sexual relations around minors" and bars public discussion of gay rights and relationships within earshot of children. "There are no legal precedents for this law, so we just don\'t know what comes under this law and (what) doesn\'t," a TV Rain spokesperson told CNN. "Therefore, fearing to break the law especially given']
histories = ['']

for j in range(3):
    wiz_says = trainer.generate_wiz_response(topics, histories, documents)
    histories = trainer.update_histories(wiz_says, histories, reverse=args.reverse)
    if not j == trainer.args.num_turns - 1:
        app_says = trainer.generate_app_response(histories)
        histories = trainer.update_histories(app_says, histories, reverse=args.reverse)

uttrs = histories[0].split(' / ')
uttrs.reverse()

with open(target_file, 'a') as f:
    _ = f.write('Topic:\t%s\n' % topics[0])
    _ = f.write('Document:\t%s\nHistory:\n' % documents[0])
    for j, uttr in enumerate(uttrs):
        if j % 2 == 0:    # Wiz
            _ = f.write('\tTeacher - %s\n' % uttr)
        else:
            _ = f.write('\tStudent - %s\n' % uttr)
    _ = f.write('\n')



'''
1.8
'''
version = 'wiz-1.8.3'
wiz_path = '../Talk_/saved_models/%s/step_9900/pytorch_model.bin' % version


with open(BASE_PATH + 'saved_models/%s/args.pkl' % version, 'rb') as f:
    args = pickle.load(f)

with open(BASE_PATH + 'za/args/args_doha_train.pkl', 'rb') as f:
    args_wiz = pickle.load(f)
    args_wiz.experiment_type = 'chat_document'
    args_wiz.model_file_path = args.wiz_path
    args_wiz.model_name = args.wiz_model_name
    wiz = MultiBartQA(args_wiz, device)
    if wiz_path != None:
        wiz.generator.load_state_dict(torch.load(wiz_path))

with open(BASE_PATH + 'za/args/args_bart_train.pkl', 'rb') as f:
    args_app = pickle.load(f)
    args_app.experiment_type = 'chat_document'
    args_app.model_file_path = '../Talk_/saved_models/app-1.1.1/checkpoint-29000/model.pt'
    args_app.model_name = args.app_model_name
    app = BartQA(args_app, device)

scorer_cov = CoverageScorer()
# coherence scorer
with open(BASE_PATH + 'za/args/args_coh.pkl', 'rb') as f:
    args_coh = pickle.load(f)
    args_coh.model_name_or_path = '../Talk_/saved_models/coh-1.0/pytorch_model.bin'

scorer_coh = CoherenceScorer(args_coh, accelerator.device)
scorers = [scorer_cov, scorer_coh]

wiz, app = accelerator.prepare(wiz, app)
alphas = [args.alpha_cov, args.alpha_coh]
trainer = RLTrainerForGenerator(args, wiz, app, scorers, alphas, accelerator)

topics = ['Consistency-based anomaly detection with adaptive multiple-hypotheses predictions']
documents = ['In one-class-learning tasks, only the normal case can be modeled with data, whereas the variation of all possible anomalies is too large to be described sufficiently by samples. Thus, due to the lack of representative data, the wide-spread discriminative approaches cannot cover such learning tasks, and rather generative models, which attempt to learn the input density of the normal cases, are used. However, generative models suffer from a large input dimensionality (as in images) and are typically inefficient learners. We propose to learn the data distribution more efficiently with a multi-hypotheses autoencoder. Moreover, the model is criticized by a discriminator, which prevents artificial data modes not supported by data, and which enforces diversity across hypotheses. This consistency-based anomaly detection (ConAD) framework allows the reliable identification of outof- distribution samples. For anomaly detection on CIFAR-10, it yields up to 3.9% points improvement over previously reported results. On a real anomaly detection task, the approach reduces the error of the baseline models from 6.8% to 1.5%.']
histories = ['']

for j in range(3):
    wiz_says = trainer.generate_wiz_response(topics, histories, documents)
    histories = trainer.update_histories(wiz_says, histories, reverse=args.reverse)
    if not j == trainer.args.num_turns - 1:
        app_says = trainer.generate_app_response(histories)
        histories = trainer.update_histories(app_says, histories, reverse=args.reverse)

uttrs = histories[0].split(' / ')
uttrs.reverse()

with open(target_file, 'a') as f:
    _ = f.write('Topic:\t%s\n' % topics[0])
    _ = f.write('Document:\t%s\nHistory:\n' % documents[0])
    for j, uttr in enumerate(uttrs):
        if j % 2 == 0:    # Wiz
            _ = f.write('\tTeacher - %s\n' % uttr)
        else:
            _ = f.write('\tStudent - %s\n' % uttr)
    _ = f.write('\n')



'''
1.9
'''
version = 'wiz-1.8.3'
wiz_path = '../Talk_/saved_models/%s/step_9900/pytorch_model.bin' % version


with open(BASE_PATH + 'saved_models/%s/args.pkl' % version, 'rb') as f:
    args = pickle.load(f)

with open(BASE_PATH + 'za/args/args_doha_train.pkl', 'rb') as f:
    args_wiz = pickle.load(f)
    args_wiz.experiment_type = 'chat_document'
    args_wiz.model_file_path = args.wiz_path
    args_wiz.model_name = args.wiz_model_name
    wiz = MultiBartQA(args_wiz, device)

with open(BASE_PATH + 'za/args/args_bart_train.pkl', 'rb') as f:
    args_app = pickle.load(f)
    args_app.experiment_type = 'chat_document'
    args_app.model_file_path = '../Talk_/saved_models/app-1.1.1/checkpoint-29000/model.pt'
    args_app.model_name = args.app_model_name
    app = BartQA(args_app, device)

scorer_cov = CoverageScorer()
# coherence scorer
with open(BASE_PATH + 'za/args/args_coh.pkl', 'rb') as f:
    args_coh = pickle.load(f)
    args_coh.model_name_or_path = '../Talk_/saved_models/coh-1.0/pytorch_model.bin'

scorer_coh = CoherenceScorer(args_coh, accelerator.device)
scorers = [scorer_cov, scorer_coh]

wiz, app = accelerator.prepare(wiz, app)
alphas = [args.alpha_cov, args.alpha_coh]
trainer = RLTrainerForGenerator(args, wiz, app, scorers, alphas, accelerator)

topics = ['You had progressive worsening of a skin infection']
documents = ['Your cellulitis (skin infection) was treated with an intra venous antibiotic called ceftriaxone.  You got 4 days worth of this antibiotic while in hospital.  Your leg was elevated and you were given Tylenol as needed for pain.  Your symptoms improved and you were transitioned to an oral antibiotics called vantin 400mg by mouth twice a day to complete a total of 10 days of antibiotics.']
histories = ['']

for j in range(3):
    wiz_says = trainer.generate_wiz_response(topics, histories, documents)
    histories = trainer.update_histories(wiz_says, histories, reverse=args.reverse)
    if not j == trainer.args.num_turns - 1:
        app_says = trainer.generate_app_response(histories)
        histories = trainer.update_histories(app_says, histories, reverse=args.reverse)

uttrs = histories[0].split(' / ')
uttrs.reverse()

with open(target_file, 'a') as f:
    _ = f.write('Topic:\t%s\n' % topics[0])
    _ = f.write('Document:\t%s\nHistory:\n' % documents[0])
    for j, uttr in enumerate(uttrs):
        if j % 2 == 0:    # Wiz
            _ = f.write('\tTeacher - %s\n' % uttr)
        else:
            _ = f.write('\tStudent - %s\n' % uttr)
    _ = f.write('\n')


topics = ['After visit summary']
documents = ["You were hospitalized because: 1. Syncope due to orthostatic hypotension and infection    Summary: you were admitted due to syncope due to orthostatic hypotension and infection. You were found to have on CT abdomen imagining to have a presacral fluid collection about 8 cm in size. You were treated with iv fluids and iv antibiotics and will need to have the fluid collection drained. You were seen by general surgery and the recommendation is for you to be transferred to St. [** PERSON **]' s where you had your previous surgery for continued care and management."]
histories = ['']

for j in range(3):
    wiz_says = trainer.generate_wiz_response(topics, histories, documents)
    histories = trainer.update_histories(wiz_says, histories, reverse=args.reverse)
    if not j == trainer.args.num_turns - 1:
        app_says = trainer.generate_app_response(histories)
        histories = trainer.update_histories(app_says, histories, reverse=args.reverse)

uttrs = histories[0].split(' / ')
uttrs.reverse()

with open(target_file, 'a') as f:
    _ = f.write('Topic:\t%s\n' % topics[0])
    _ = f.write('Document:\t%s\nHistory:\n' % documents[0])
    for j, uttr in enumerate(uttrs):
        if j % 2 == 0:    # Wiz
            _ = f.write('\tTeacher - %s\n' % uttr)
        else:
            _ = f.write('\tStudent - %s\n' % uttr)
    _ = f.write('\n')










# Get test articles
path_test = '../Talk_/data/WoW-selector-1.5/test.json'
# path_test = '../Talk_/data/CNN-DailyMail-processed/test.json'
# path_test = '../Talk_/data/Papers-processed/test.json'
# path_test = '../avs/data/json/src-tgt_1.0/test.json'

with open(path_test) as f:
    con = json.load(f)['data']

