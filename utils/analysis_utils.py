import re

pat_dtype = re.compile("dtype = \'(.+)\'")
pat_idx = re.compile("idx = ([0-9]+)")
pat_models = re.compile("make_human_eval_conversation\(trainer_(.), topic, doc")
pat_qna = re.compile("QA correct answers: ([0-9])\/([0-9])")
pat_coh = re.compile("Average Coherence score: ([\.0-9]+)")
pat_rdb = re.compile("Average Readability Score: ([\.0-9]+)")
pat_ova = re.compile("Overall score: ([0-3])")
