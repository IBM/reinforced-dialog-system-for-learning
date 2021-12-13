'''
This script evaluates the robot talk results when
    A selector is used to select from 16 candidate responses
'''
import re
import numpy as np
import matplotlib.pyplot as plt


path_in = '../Talk_/logs/self_play_rl_1.5.6.txt'

with open(path_in) as f:
    con = f.read()

'''
Analyze eval set
'''

pat_cov = re.compile("--- Eval average coverage score: ([0-9\.]+) ---")
pat_coh = re.compile("--- Eval average coherence score: ([0-9\.]+) ---")
pat_eval_start = re.compile("--- Eval epoch [0-9]+ starts ---")
pat_eval_end = re.compile("--- Eval time consumed: ")
pat_reward = re.compile("Reward:\t([0-9\.]+)\n")

rewards_cov = []
start = 0
rewards_cov = [float(score) for score in pat_cov.findall(con)]
rewards_coh = [float(score) for score in pat_coh.findall(con)]


assert len(rewards_coh) == len(rewards_cov)
epoches = [(i+1)*600 for i in range(len(rewards_cov))]

l1=plt.plot(epoches, rewards_cov, 'r--', label='coverage rewards average')
l2=plt.plot(epoches, rewards_coh, 'b--', label='coherence rewards average')

plt.title('Rewards every 600 steps')
plt.xlabel('Steps')
plt.ylabel('Rewards')
plt.legend()
plt.show()
#
# '''
# Analyze train set
# '''
# pat_sample = re.compile("Sample reward:\tCoverage - ([0-9\.]+)\tCoherence - ([0-9\.]+)")
# pat_base = re.compile("Baseline reward:\tCoverage - ([0-9\.]+)\tCoherence - ([0-9\.]+)")
#
# rewards_sample = pat_sample.findall(con)
# rewards_base = pat_sample.findall(con)
# assert len(rewards_sample) == len(rewards_base)
#
# reward_sample_cov = [float(reward[0]) for reward in rewards_sample]
# reward_sample_coh = [float(reward[1]) for reward in rewards_sample]
# reward_base_cov = [float(reward[0]) for reward in rewards_base]
# reward_base_coh = [float(reward[1]) for reward in rewards_base]
#
# reward_sample_cov_avg = [np.mean(reward_sample_cov[i:i+100]) for i in range(len(reward_sample_cov)-100)]
# reward_sample_coh_avg = [np.mean(reward_sample_coh[i:i+100]) for i in range(len(reward_sample_coh)-100)]
# reward_base_cov_avg = [np.mean(reward_base_cov[i:i+100]) for i in range(len(reward_base_cov)-100)]
# reward_base_coh_avg = [np.mean(reward_base_coh[i:i+100]) for i in range(len(reward_base_coh)-100)]
# steps = list(range(len(reward_base_coh_avg)))
#
# l_sample_cov=plt.plot(steps, reward_sample_cov_avg, 'r--', label='coverage rewards-sample')
# l_sample_coh=plt.plot(steps, reward_sample_coh_avg, 'b--', label='coherence rewards-sample')
#
# plt.title('Rewards every step steps')
# plt.xlabel('Steps')
# plt.ylabel('Rewards')
# plt.legend()
# plt.show()
#
# l_base_cov=plt.plot(steps, reward_sample_cov_avg, 'r--', label='coverage rewards sample')
# l_base_coh=plt.plot(steps, reward_sample_coh_avg, 'b--', label='coherence rewards sample')
# plt.title('Rewards every step steps')
# plt.xlabel('Steps')
# plt.ylabel('Rewards')
# plt.legend()
# plt.show()


