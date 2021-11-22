import re

pat = re.compile("--- Eval average coverage score: ([\.0-9]+) ---")

path_in = '../Talk_/logs/self_play_rl_1.5.7.txt'

with open(path_in) as f:
    con = f.read()

reward_scores = pat.findall(con)


