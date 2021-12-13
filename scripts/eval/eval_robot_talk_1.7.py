import re
from datasets import load_dataset
from collections import defaultdict
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np
from utils.eval_utils import SummaQAEvaluator, get_uttrs
import matplotlib.pyplot as plt

'''
This are old results
'''

data_files_rl = {
    'test': '../Talk_/data/CNN-DailyMail-processed/test.json',
}
NUM_TEST = 1000
extension = 'json'
raw_datasets_rl = load_dataset(extension, data_files=data_files_rl, field='data')
eval_dataset_rl = raw_datasets_rl['test'].select(range(NUM_TEST))
scorer_cov = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scorer_qa = SummaQAEvaluator()
pat_conv = re.compile('History: ([^\n]+)')
pat_score = re.compile('Step [0-9]+ instance [0-9] \| Cov score: ([\.0-9]+) \| Coh score: ([\.0-9]+) \|')


def eval_all(path, do_rouge=True, do_read=True, do_qa=True):
    assert path.endswith('.txt')
    with open(path) as f:
        con = f.read()
    path_out = path[:-4] + '.results.txt'
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
    with open(path_out, 'w') as f_out:
        for key, values in scores.items():
            if '-3' in key:
                continue
            print('Avg score of %s: %s' % (key, np.mean(values)))
            f_out.write('Avg score of %s: %s\n' % (key, np.mean(values)))


# A. Naive
eval_all('../Talk_/results/1.7.naive.txt')
# Avg score of rouge1-0_gain: 0.1642764392100364
# Avg score of rouge2-0_gain: 0.1240087559543887
# Avg score of rougeL-0_gain: 0.14821187464745172
# Avg score of rouge1-1_gain: 0.1299162834512313
# Avg score of rouge2-1_gain: 0.10046669140447
# Avg score of rougeL-1_gain: 0.10666507472417243
# Avg score of rouge1-2_gain: 0.09476885474713657
# Avg score of rouge2-2_gain: 0.07647850391950763
# Avg score of rougeL-2_gain: 0.0742575053989034
# Avg score of rouge1: 0.38896157740840426
# Avg score of rouge2: 0.30095395127836627
# Avg score of rougeL: 0.32913445477052755
# Avg score of cov: 0.389538
# Avg score of coh: 0.539
# Avg score of avg_prob: 0.15902350828000855
# Avg score of avg_fscore: 0.15802401430641164
# Avg score of len-0: 15.765
# Avg score of len-1: 17.296
# Avg score of len-2: 15.829
# Avg score of len: 48.89 / 16.29

# B. Cov only
eval_all('../Talk_/results/1.7.cov.txt')
# Avg score of rouge1-0_gain: 0.3778705671246746
# Avg score of rouge2-0_gain: 0.36016245337894287
# Avg score of rougeL-0_gain: 0.37133442348600815
# Avg score of rouge1-1_gain: 0.27370500868826436
# Avg score of rouge2-1_gain: 0.2650318618851282
# Avg score of rougeL-1_gain: 0.25195167589162243
# Avg score of rouge1-2_gain: 0.15703471354270476
# Avg score of rouge2-2_gain: 0.15275138262351623
# Avg score of rougeL-2_gain: 0.11167307194121198
# Avg score of rouge1: 0.8086102893556436
# Avg score of rouge2: 0.7779456978875874
# Avg score of rougeL: 0.7349591713188426
# Avg score of cov: 0.81295
# Avg score of coh: 0.40476699999999993
# Avg score of avg_prob: 0.312425904912784
# Avg score of avg_fscore: 0.41118046705559
# Avg score of len-0: 34.273
# Avg score of len-1: 38.211
# Avg score of len-2: 31.604
# Avg score of len: 104.08 / 34.69

# C. Coh only
eval_all('../Talk_/results/1.7.coh.txt')

# Avg score of rouge1-0_gain: 0.1600608654183648
# Avg score of rouge2-0_gain: 0.11593439290423721
# Avg score of rougeL-0_gain: 0.14201893620441705
# Avg score of rouge1-1_gain: 0.11825462288084071
# Avg score of rouge2-1_gain: 0.08071781426409157
# Avg score of rougeL-1_gain: 0.09206164179131679
# Avg score of rouge1-2_gain: 0.07862299974696822
# Avg score of rouge2-2_gain: 0.05610730122856155
# Avg score of rougeL-2_gain: 0.05836711652236551
# Avg score of rouge1: 0.3569384880461737
# Avg score of rouge2: 0.2527595083968903
# Avg score of rougeL: 0.29244769451809943
# Avg score of cov: 0.35769999999999996
# Avg score of coh: 0.6414
# Avg score of avg_prob: 0.14954897520199573
# Avg score of avg_fscore: 0.1407143018227219
# Avg score of len-0: 15.88
# Avg score of len-1: 17.232
# Avg score of len-2: 15.59
# Avg score of len: 48.70 / 16.23


# D. Full
eval_all('../Talk_/results/1.7.full.txt')

# Avg score of rouge1-0_gain: 0.3468106477490993
# Avg score of rouge2-0_gain: 0.3272448043200315
# Avg score of rougeL-0_gain: 0.33912974463988804
# Avg score of rouge1-1_gain: 0.2358594944975245
# Avg score of rouge2-1_gain: 0.223627257031156
# Avg score of rougeL-1_gain: 0.21926393263743818
# Avg score of rouge1-2_gain: 0.14608861020781497
# Avg score of rouge2-2_gain: 0.1376275298602181
# Avg score of rougeL-2_gain: 0.11797660308535582
# Avg score of rouge1: 0.7275900435727762
# Avg score of rouge2: 0.6873985709725239
# Avg score of rougeL: 0.6754264675379992
# Avg score of cov: 0.7356739999999999
# Avg score of coh: 0.45089999999999997
# Avg score of avg_prob: 0.28320244262788574
# Avg score of avg_fscore: 0.36076620423985845
# Avg score of len-0: 33.424
# Avg score of len-1: 34.593
# Avg score of len-2: 29.61391129032258
# Avg score of len: 97.63 / 32.54


turns = np.array([1, 2, 3], dtype=int)
gain_1 = [39.45, 18.71, 11.25]
gain_2 = [37.36, 17.92, 10.46]
gain_L = [38.27, 9.34, 3.88]
lens = [38.83, 30.32, 25.38]

fig, ax1 = plt.subplots()
ax1.set_xlabel('Turn')
ax1.set_ylabel('Rouge Score')
ax1.bar(turns - 0.2, gain_1, width=0.2, color='#009DAE', align='center', label='Rouge-1')
ax1.bar(turns, gain_2, width=0.2, color='#71DFE7', align='center', label='Rouge-2')
ax1.bar(turns + 0.2, gain_L, width=0.2, color='#C2FFF9', align='center', label='Rouge-L')
plt.xticks(np.arange(min(turns), max(turns) + 1, 1))
# ax1.legend(loc=0)
# ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
ax2.set_ylabel('Number of words per utterence')  # we already handled the x-label with ax1
ax2.set_ylim([0, 40])
ax2.plot(turns, lens, 'go--', color='#FFE652', label='Number of words')
# ax2.legend(loc=0)

# ask matplotlib for the plotted objects and their labels
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
# fig.set_size_inches(5.1, 3.9)
plt.savefig('../Talk_/imgs/rouge_gain_1.8.3.pdf', dpi=180)
plt.show()
