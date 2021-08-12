'''
This script evaluates the robot talk results when
    A selector is used to select from 16 candidate responses
'''
import re
import numpy as np
import matplotlib.pyplot as plt

pat = re.compile("--- Eval average reward: ([0-9\.]+) ---")

path_in = '../Talk_/logs/logs_1.5.txt'

with open(path_in) as f:
    con = f.read()

rewards = []
start = 0
se = pat.search(con, start)
while se:
    rewards.append(float(se.group(1)))
    start = se.end()
    se = pat.search(con, start)

rewards_last_three = [np.mean(rewards[max(i-3, 0):i+1]) for i in range(len(rewards))]
epoches = [(i+1)*600 for i in range(len(rewards))]

l1=plt.plot(epoches, rewards, 'r--', label='rewards')
l2=plt.plot(epoches, rewards_last_three, 'b--', label='average rewards last three times')

plt.title('Rouge-2 F1 score on validation set')
plt.xlabel('Steps')
plt.ylabel('Rewards')
plt.legend()
plt.show()

