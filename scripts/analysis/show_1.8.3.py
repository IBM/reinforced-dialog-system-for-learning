import re
import matplotlib
import matplotlib.pyplot as plt

versions = ['1.8.3.1', '1.8.3.2', '1.8.3']
pat_cov = re.compile('--- Eval average coverage score: ([0-9\.]+) ---')
pat_coh = re.compile('--- Eval average coherence score: ([0-9\.]+) ---')

cov_scores_all = []
coh_scores_all = []
for version in versions:
    path_in = '../Talk_/logs/self_play_rl_%s.txt' % version
    with open(path_in) as f:
        con = f.read()
    cov_scores = [float(score) for score in pat_cov.findall(con)]
    coh_scores = [float(score) for score in pat_coh.findall(con)]
    steps = list([300 * (i + 1) for i in range(len(cov_scores))])
    cov_scores_all.append(cov_scores)
    coh_scores_all.append(coh_scores)


font = {'size': 16}
matplotlib.rc('font', **font)

f = plt.figure(figsize=(16,3.9))
ax1 = f.add_subplot(131)
plt.plot(steps, cov_scores_all[0], color="#FB3640")
plt.plot(steps, coh_scores_all[0], color="#247BA0")
plt.ylabel('Coverage / Coherence Score')
# plt.tick_params('y', labelbottom=False)
plt.title('(a) +Coverage Only')

# share x only
ax2 = f.add_subplot(132, sharey=ax1)
plt.plot(steps, cov_scores_all[1], color="#FB3640")
plt.plot(steps, coh_scores_all[1], color="#247BA0")
# make these tick labels invisible
plt.tick_params('y', labelleft=False)
plt.xlabel('Steps')
plt.title('(b) +Coherence Only')

# share x and y
ax3 = f.add_subplot(133, sharey=ax1)
plt.plot(steps, cov_scores_all[2], color="#FB3640", label='coverage')
plt.plot(steps, coh_scores_all[2], color="#247BA0", label='coherence')
# make these tick labels invisible
plt.tick_params('y', labelleft=False)
plt.legend(loc='lower right')
plt.title('(c) +Coherence & Coherence')

plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.2)

plt.savefig('../Talk_/imgs/scores.pdf')
plt.show()


