import re
import matplotlib.pyplot as plt

version = '1.7.1'
path_in = '../Talk_/logs/self_play_rl_%s.txt' % version
with open(path_in) as f:
    con = f.read()

pat_cov = re.compile('--- Eval average coverage score: ([0-9\.]+) ---')
pat_coh = re.compile('--- Eval average coherence score: ([0-9\.]+) ---')

cov_scores = [float(score) for score in pat_cov.findall(con)]
coh_scores = [float(score) for score in pat_coh.findall(con)]
steps = list([300 * i for i in range(len(cov_scores))])

l1 = plt.plot(steps, cov_scores, 'r-', label='coverage')
l2 = plt.plot(steps, coh_scores, 'b-', label='coherence')
plt.title('Coverage & Coherence scores every 300 steps')
plt.xlabel('Steps')
plt.ylabel('Scores')
plt.title(version)
plt.legend()
plt.show()



