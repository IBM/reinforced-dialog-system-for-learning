import random
import openreview
import requests
import re
import os
from tqdm import tqdm
import json

instances = []
# ICLR (10533)
client = openreview.Client(baseurl='https://api.openreview.net', username='pengshancai@umass.edu', password='19911017Cps')
invitations = {
    "ICLR.cc/2018/Conference/-/Blind_Submission",
    "ICLR.cc/2019/Conference/-/Blind_Submission",
    "ICLR.cc/2020/Conference/-/Blind_Submission",
    "ICLR.cc/2021/Conference/-/Blind_Submission",
    "ICLR.cc/2022/Conference/-/Blind_Submission",
}
cnt = 0
for invitation in invitations:
    cnt_sub = 0
    venue = invitation.split('/')[0]
    notes = openreview.tools.iterget_notes(client, invitation=invitation)
    for note in notes:
        if 'abstract' not in note.content:
            continue
        if len(note.content['title']) > 10 and len(note.content['abstract']) > 10:
            cnt_sub += 1
            instances.append({
                'topic': note.content['title'],
                'document': note.content['abstract']
            })
    cnt += cnt_sub
    print('%s:\t%s' % (invitation, cnt_sub))


# ACL Anthology (14273)
# Step 1: Download Dataset (This part is locked as the file has been generated)
# conf_urls = [
#     "https://aclanthology.org/events/acl-2021/",
#     "https://aclanthology.org/events/acl-2020/",
#     "https://aclanthology.org/events/acl-2019/",
#     "https://aclanthology.org/events/acl-2018/",
#     "https://aclanthology.org/events/acl-2017/",
#     "https://aclanthology.org/events/emnlp-2020/",
#     "https://aclanthology.org/events/emnlp-2019/",
#     "https://aclanthology.org/events/emnlp-2018/",
#     "https://aclanthology.org/events/emnlp-2017/",
#     "https://aclanthology.org/events/naacl-2021/",
#     "https://aclanthology.org/events/naacl-2019/",
#     "https://aclanthology.org/events/naacl-2018/",
#     "https://aclanthology.org/events/eacl-2021/",
#     "https://aclanthology.org/events/eacl-2017/",
#     "https://aclanthology.org/events/findings-2021/",
#     "https://aclanthology.org/events/findings-2020/"
# ]
# pat_paper_url = re.compile("<a class=align-middle href=([^>]+)>")
# out_file = '../Talk_/data/Papers-raw/acl_papers.txt'
# if not os.path.exists(out_file):
#     with open(out_file, 'w') as f:
#         f.write('')
#
#
# def get_paper_list(conf_url):
#     index_html = requests.get(conf_url).text
#     paper_urls = pat_paper_url.findall(index_html)
#     return paper_urls
#
#
# def get_paper_info(paper_url):
#     title, abstract = None, None
#     try:
#         paper_url_full = "https://aclanthology.org" + paper_url
#         paper_html = requests.get(paper_url_full).text
#         info_start = paper_html.find("<pre id=citeEndnoteContent class=\"bg-light border p-2\" style=max-height:50vh>")
#         info_end = paper_html.find("</pre>", info_start)
#         info = paper_html[info_start: info_end].split('\n')
#         for line in info:
#             if line.startswith("%T"):
#                 title = line.replace('%T', '').strip()
#             if line.startswith('%X'):
#                 abstract = line.replace('%X', '').strip()
#     except:
#         print('Error @ %s ' % paper_url)
#     return title, abstract
#
#
# def write_paper_info(paper_url, title, abstract):
#     with open(out_file, 'a') as f:
#         _ = f.write('%s\t%s\t%s\n' % (paper_url, title, abstract))
#
#
# paper_urls = []
# for conf_url in conf_urls:
#     paper_urls += get_paper_list(conf_url)
#
# print('We have %s candidate papers' % len(paper_urls))
#
# progress_bar = tqdm(range(len(paper_urls)))
# for i, paper_url in enumerate(paper_urls):
#     title, abstract = get_paper_info(paper_url)
#     write_paper_info(paper_url, title, abstract)
#     progress_bar.update(1)

# Step 2: Convert the dataset into a certain format

# with open('../Talk_/data/Papers-raw/acl_papers.txt') as f:
#     con = f.readlines()
#
# for line in con:
#     _, title, document = line.strip().split('\t')
#     if title != 'None' and document != 'None':
#         instances.append({
#             'topic': title,
#             'document': document
#         })
#
# random.shuffle(instances)
# out_path_processed = '../Talk_/data/Papers-processed/'
# if not os.path.exists(out_path_processed):
#     os.mkdir(out_path_processed)
#
# with open(out_path_processed + 'train.json', 'w') as f:
#     dataset = {
#         'version': 1.8,
#         'data': instances[:-1000]
#     }
#     json.dump(dataset, f)
#
# with open(out_path_processed + 'dev.json', 'w') as f:
#     dataset = {
#         'version': 1.8,
#         'data': instances[-1000: -500]
#     }
#     json.dump(dataset, f)
#
# with open(out_path_processed + 'test.json', 'w') as f:
#     dataset = {
#         'version': 1.8,
#         'data': instances[-500:]
#     }
#     json.dump(dataset, f)
