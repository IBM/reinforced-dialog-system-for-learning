import re
from datasets import load_dataset
from collections import defaultdict
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np
from utils.eval_utils import SummaQAEvaluator, get_uttrs

data_files_rl = {
    'test': '../Talk_/data/Papers-processed/test.json',
}
extension = 'json'
raw_datasets_rl = load_dataset(extension, data_files=data_files_rl, field='data')
eval_dataset_rl = raw_datasets_rl['test']
scorer_cov = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scorer_qa = SummaQAEvaluator()
pat_conv = re.compile('History: ([^\n]+)')
pat_score = re.compile('Step [0-9]+ instance [0-9] \| Cov score: ([\.0-9]+) \| Coh score: ([\.0-9]+) \|')


def eval_all(con, do_rouge=True, do_read=True, do_qa=True):
    print('Eval gain in each turn')
    convs = pat_conv.findall(con)
    scores_ = pat_score.findall(con)
    scores = defaultdict(list)
    assert len(scores_) == len(convs)
    # 1.1 Gain in each turn
    if do_rouge:
        progress = tqdm(range(len(convs)))
        for i, conv in enumerate(convs):
            rec = eval_dataset_rl[i]
            document = rec['document']
            uttrs_wiz, uttrs_app, uttrs = get_uttrs(conv)
            history = ''
            assert len(uttrs_wiz) > 1
            for j, uttr in enumerate(uttrs_wiz):
                if j > 2:
                    continue
                len_uttr = len(uttr.split(' '))
                scores['len-%s' % j].append(len_uttr)
                score_old = scorer_cov.score(document, history)
                history += uttr + ' '
                score_new = scorer_cov.score(document, history)
                for key in score_new.keys():
                    gain = score_new[key].fmeasure - score_old[key].fmeasure
                    scores['%s-%s_gain' % (key, j)].append(gain)
            for key in score_new.keys():
                scores[key].append(score_new[key].fmeasure)
            _ = progress.update(1)
    # 1.2 Average score (cov, coh)
    if do_read:
        print('Eval average score (cov, coh)')
        for cov, coh in scores_:
            scores['cov'].append(float(cov))
            scores['coh'].append(float(coh))
    # 1.3 QA scores
    if do_qa:
        print('Eval QA')
        progress = tqdm(range(len(convs)))
        for i, conv in enumerate(convs):
            rec = eval_dataset_rl[i]
            article = rec['document']
            scores_qa = scorer_qa.get_conv_score(conv, article)
            for key, value in scores_qa.items():
                scores[key].append(value)
            _ = progress.update(1)
    # for level in ['1', '2', 'L']:
    #     for turn in [0, 1, 2]:
    #         print('Turn %s rouge%s gain: %s' % (turn, level, np.mean(scores['rouge%s-%s_gain' % (level, turn)])))
    for key, values in scores.items():
        # if key.startswith('rouge'):
        #     continue
        if '-3' in key:
            continue
        print('Avg score of %s: %s' % (key, np.mean(values)))


# A. Naive
with open('../Talk_/results/1.8.naive.txt') as f:
    con = f.read()

eval_all(con, True, False, False)
# Avg score of rouge1-0_gain: 0.14191488412735853
# Avg score of rouge2-0_gain: 0.10664595350681047
# Avg score of rougeL-0_gain: 0.12768305418124054
# Avg score of rouge1-1_gain: 0.12281597994754054
# Avg score of rouge2-1_gain: 0.09479810136200077
# Avg score of rougeL-1_gain: 0.09631789050727312
# Avg score of rouge1-2_gain: 0.0979488472580667
# Avg score of rouge2-2_gain: 0.08059147282107691
# Avg score of rougeL-2_gain: 0.0820654493211945
# Avg score of rouge1: 0.362700490553745
# Avg score of rouge2: 0.28203208771740795
# Avg score of rougeL: 0.3060885138253764
# Avg score of cov: 0.36386
# Avg score of coh: 0.4814
# Avg score of avg_prob: 0.10296706764204708
# Avg score of avg_fscore: 0.05342810707543444
# Avg score of len-0: 14.74
# Avg score of len-1: 16.438
# Avg score of len-2: 15.664
# Avg score of len: 46.84

# B. Cov only
with open('../Talk_/results/1.8.cov.txt') as f:
    con = f.read()

eval_all(con, True, False, False)
# Avg score of rouge1-0_gain: 0.37353823614643755
# Avg score of rouge2-0_gain: 0.3503365972067465
# Avg score of rougeL-0_gain: 0.36764303152884936
# Avg score of rouge1-1_gain: 0.21750276725346715
# Avg score of rouge2-1_gain: 0.21349061145948797
# Avg score of rougeL-1_gain: 0.10814624549638918
# Avg score of rouge1-2_gain: 0.13535635845771432
# Avg score of rouge2-2_gain: 0.13317094417685804
# Avg score of rougeL-2_gain: 0.01584102036041438
# Avg score of rouge1: 0.7263973618576189
# Avg score of rouge2: 0.6969981528430925
# Avg score of rougeL: 0.49163029738565295
# Avg score of cov: 0.7420040000000001
# Avg score of coh: 0.1436
# Avg score of avg_prob: 0.20321843063944237
# Avg score of avg_fscore: 0.17619810677127584
# Avg score of len-0: 36.706
# Avg score of len-1: 35.146
# Avg score of len-2: 28.804
# Avg score of len: 100.656 / 33.552

# C. Coh only
with open('../Talk_/results/1.8.coh.txt') as f:
    con = f.read()

eval_all(con, True, False, False)
# Avg score of rouge1-0_gain: 0.13193825357240113
# Avg score of rouge2-0_gain: 0.09252368561977753
# Avg score of rougeL-0_gain: 0.11347906490203975
# Avg score of rouge1-1_gain: 0.11821143625781716
# Avg score of rouge2-1_gain: 0.08091192319290484
# Avg score of rougeL-1_gain: 0.09110407106871937
# Avg score of rouge1-2_gain: 0.06945727018275583
# Avg score of rouge2-2_gain: 0.04351557537521698
# Avg score of rougeL-2_gain: 0.05242944291475386
# Avg score of rouge1: 0.3196069600129741
# Avg score of rouge2: 0.21695118418789935
# Avg score of rougeL: 0.257012578885513
# Avg score of cov: 0.320576
# Avg score of coh: 0.5438
# Avg score of avg_prob: 0.09475028950349638
# Avg score of avg_fscore: 0.043091415225185656
# Avg score of len-0: 15.35
# Avg score of len-1: 17.58
# Avg score of len-2: 15.98
# Avg score of len: 48.91


# D. Full
with open('../Talk_/results/1.8.full.txt') as f:
    con = f.read()

eval_all(con, True, False, False)
# Avg score of rouge1-0_gain: 0.39453509331696235
# Avg score of rouge2-0_gain: 0.37360122460196515
# Avg score of rougeL-0_gain: 0.38275424787503426
# Avg score of rouge1-1_gain: 0.18717967170517777
# Avg score of rouge2-1_gain: 0.17924771176094959
# Avg score of rougeL-1_gain: 0.09348789711459929
# Avg score of rouge1-2_gain: 0.11256542095759016
# Avg score of rouge2-2_gain: 0.1046101572626297
# Avg score of rougeL-2_gain: 0.038832281414380526
# Avg score of rouge1: 0.6933796626120696
# Avg score of rouge2: 0.6566222123674433
# Avg score of rougeL: 0.5147637681526991
# Avg score of len-0: 38.836
# Avg score of len-1: 30.328
# Avg score of len-2: 25.383064516129032
# Avg score of cov: 0.705346
# Avg score of coh: 0.43679999999999997
# Avg score of avg_prob: 0.17962882815205286
# Avg score of avg_fscore: 0.1845763328746519

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {'size': 12}

matplotlib.rc('font', **font)
turns = np.array([1, 2, 3], dtype=int)
gain_1 = [39.45, 18.71, 11.25]
gain_2 = [37.36, 17.92, 10.46]
gain_L = [38.27, 9.34,  3.88]
lens = [38.83, 30.32, 25.38]

fig, ax1 = plt.subplots()
ax1.set_xlabel('Turn')
ax1.set_ylabel('Information Gain')
ax1.bar(turns-0.2,  gain_1, width=0.2, color='#009DAE', align='center', label='Rouge-1')
ax1.bar(turns,      gain_2, width=0.2, color='#71DFE7', align='center', label='Rouge-2')
ax1.bar(turns+0.2,  gain_L, width=0.2, color='#C2FFF9', align='center', label='Rouge-L')
plt.xticks(np.arange(min(turns), max(turns)+1, 1))
# ax1.legend(loc=0)
# ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
ax2.set_ylabel('Number of Words')  # we already handled the x-label with ax1
ax2.set_ylim([0, 50])
ax2.plot(turns, lens, 'go--', color='#FFE652', label='Utterance Length')
# ax2.legend(loc=0)

# ask matplotlib for the plotted objects and their labels
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.set_size_inches(6.0, 3.5)
plt.savefig('../Talk_/imgs/rouge_gain_1.8.3.pdf', dpi=180)
plt.show()
