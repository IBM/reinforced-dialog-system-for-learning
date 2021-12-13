from utils.eval_utils import eval_objective

# data_files_rl = {
#     'test': '../Talk_/data/WoW-selector-1.5/test.json',
# }
# NUM_TEST = 1000
# extension = 'json'
# raw_datasets_rl = load_dataset(extension, data_files=data_files_rl, field='data')
# eval_dataset_rl = raw_datasets_rl['test'].select(range(NUM_TEST))
# scorer_cov = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
# scorer_qa = SummaQAEvaluator()
# pat_conv = re.compile('History: ([^\n]+)')
# pat_score = re.compile('Step [0-9]+ instance [0-9] \| Cov score: ([\.0-9]+) \| Coh score: ([\.0-9]+) \|')
#
#
# def eval_all(path, do_rouge=True, do_read=True, do_qa=True):
#     assert path.endswith('.txt')
#     with open(path) as f:
#         con = f.read()
#     path_out = path[:-4] + '.results.txt'
#     print('Eval gain in each turn')
#     convs = pat_conv.findall(con)
#     scores_ = pat_score.findall(con)
#     scores = defaultdict(list)
#     assert len(scores_) == len(convs)
#     # 1.1 Gain in each turn
#     if do_rouge:
#         progress = tqdm(range(len(convs)))
#         for i, conv in enumerate(convs):
#             rec = eval_dataset_rl[i]
#             document = rec['document']
#             uttrs_wiz, uttrs_app, uttrs = get_uttrs(conv)
#             history = ''
#             assert len(uttrs_wiz) > 1
#             for j, uttr in enumerate(uttrs_wiz):
#                 if j > 2:
#                     continue
#                 len_uttr = len(uttr.split(' '))
#                 scores['len-%s' % j].append(len_uttr)
#                 score_old = scorer_cov.score(document, history)
#                 history += uttr + ' '
#                 score_new = scorer_cov.score(document, history)
#                 for key in score_new.keys():
#                     gain = score_new[key].fmeasure - score_old[key].fmeasure
#                     scores['%s-%s_gain' % (key, j)].append(gain)
#             for key in score_new.keys():
#                 scores[key].append(score_new[key].fmeasure)
#             _ = progress.update(1)
#     # 1.2 Average score (cov, coh)
#     if do_read:
#         print('Eval average score (cov, coh)')
#         for cov, coh in scores_:
#             scores['cov'].append(float(cov))
#             scores['coh'].append(float(coh))
#     # 1.3 QA scores
#     if do_qa:
#         print('Eval QA')
#         progress = tqdm(range(len(convs)))
#         for i, conv in enumerate(convs):
#             rec = eval_dataset_rl[i]
#             article = rec['document']
#             scores_qa = scorer_qa.get_conv_score(conv, article)
#             for key, value in scores_qa.items():
#                 scores[key].append(value)
#             _ = progress.update(1)
#     with open(path_out, 'w') as f_out:
#         for key, values in scores.items():
#             if '-3' in key:
#                 continue
#             print('Avg score of %s: %s' % (key, np.mean(values)))
#             f_out.write('Avg score of %s: %s\n' % (key, np.mean(values)))




NUM_TEST=1000
# A. Naive
eval_objective('../Talk_/results/1.6.naive.txt')
# Avg score of rouge1-0_gain: 0.1843811359591213
# Avg score of rouge2-0_gain: 0.15263244184553001
# Avg score of rougeL-0_gain: 0.17653399745689494
# Avg score of rouge1-1_gain: 0.16416892437118133
# Avg score of rouge2-1_gain: 0.14288261582228312
# Avg score of rougeL-1_gain: 0.1520135865436091
# Avg score of rouge1-2_gain: 0.1255743244391023
# Avg score of rouge2-2_gain: 0.1113766550707785
# Avg score of rougeL-2_gain: 0.1095106283998685
# Avg score of rouge1: 0.474124384769405
# Avg score of rouge2: 0.40689171273859165
# Avg score of rougeL: 0.4380582124003725
# Avg score of cov: 0.475507
# Avg score of coh: 0.563433
# Avg score of avg_prob: 0.20071788246090871
# Avg score of avg_fscore: 0.19474935222593445
# Avg score of len-0: 14.376
# Avg score of len-1: 16.652
# Avg score of len-2: 15.516
# Avg score of len: 46.54 / 15.51

# B. Cov only
eval_objective('../Talk_/data/WoW-selector-1.5/test.json', '../Talk_/results/1.6.9.1.txt', )
# Avg score of rouge1-0_gain: 0.32080581981666223
# Avg score of rouge2-0_gain: 0.3054801184192845
# Avg score of rougeL-0_gain: 0.3176869194805786
# Avg score of rouge1-1_gain: 0.2652854133057956
# Avg score of rouge2-1_gain: 0.25656528003261025
# Avg score of rougeL-1_gain: 0.25923046387257764
# Avg score of rouge1-2_gain: 0.16413342433873215
# Avg score of rouge2-2_gain: 0.15844117904040306
# Avg score of rougeL-2_gain: 0.149235025754112
# Avg score of rouge1: 0.7462854552770604
# Avg score of rouge2: 0.7166839891953282
# Avg score of rougeL: 0.7225707684891695
# Avg score of cov: 0.767302
# Avg score of coh: 0.42913299999999993
# Avg score of avg_prob: 0.3090243225771781
# Avg score of avg_fscore: 0.34380108280149413
# Avg score of len-0: 24.844
# Avg score of len-1: 32.3
# Avg score of len-2: 27.19877049180328
# Avg score of len: 84.334 / 28.11

# C. Coh only
eval_objective('../Talk_/data/WoW-selector-1.5/test.json', '../Talk_/results/1.6.9.2.txt', )

# D. Full 1.6.9.3 (cov 0.85, coh 0.15)
eval_objective('../Talk_/data/WoW-selector-1.5/test.json', '../Talk_/results/1.6.9.3.txt', )
# Avg score of rouge1-0_gain: 0.28163780355690377
# Avg score of rouge2-0_gain: 0.2645769837628578
# Avg score of rougeL-0_gain: 0.2786435773477035
# Avg score of rouge1-1_gain: 0.2159499786487501
# Avg score of rouge2-1_gain: 0.20386313028511643
# Avg score of rougeL-1_gain: 0.20982539777639156
# Avg score of rouge1-2_gain: 0.16738804415352226
# Avg score of rouge2-2_gain: 0.1594835936886921
# Avg score of rougeL-2_gain: 0.15443656719210916
# Avg score of rouge1: 0.6547651556658113
# Avg score of rouge2: 0.6181952085216561
# Avg score of rougeL: 0.6334849117174856
# Avg score of cov: 0.6856800000000001
# Avg score of coh: 0.4574
# Avg score of avg_prob: 0.2751654164195461
# Avg score of avg_fscore: 0.283193591218472
# Avg score of len-0: 21.258
# Avg score of len-1: 23.247
# Avg score of len-2: 23.584664536741215
# Avg score of len: 68.085 / 22.69

# E. Full 1.6.9.5 (cov 0.75, coh 0.25)
eval_objective('../Talk_/data/WoW-selector-1.5/test.json', '../Talk_/results/1.6.9.5.txt', )
# Avg score of rouge1-0_gain: 0.26156647790391685
# Avg score of rouge2-0_gain: 0.24376315191912812
# Avg score of rougeL-0_gain: 0.25857622034192873
# Avg score of rouge1-1_gain: 0.20944471928838224
# Avg score of rouge2-1_gain: 0.19689493230812227
# Avg score of rougeL-1_gain: 0.203491839912141
# Avg score of rouge1-2_gain: 0.15684435249247355
# Avg score of rouge2-2_gain: 0.14764627753670687
# Avg score of rougeL-2_gain: 0.14403787832814288
# Avg score of rouge1: 0.6278555496847725
# Avg score of rouge2: 0.5883043617639572
# Avg score of rougeL: 0.6061059385822127
# Avg score of cov: 0.634084
# Avg score of coh: 0.515633
# Avg score of avg_prob: 0.25944750048706694
# Avg score of avg_fscore: 0.27011519901092795
# Avg score of len-0: 19.146
# Avg score of len-1: 21.806
# Avg score of len-2: 21.126
# Avg score of len: 62.07 / 20.69


