import os
import json
import random

ratio = 0.1
os.system('cp ./data/wikisql_tok/train_tok.jsonl ./data/wikisql_tok/train_tok_full.jsonl')

data = []
with open('./data/wikisql_tok/train_tok_full.jsonl','r') as fin:
    for idx, line in enumerate(fin):
        t1 = json.loads(line.strip())
        data.append(t1)

dsize = int(len(data)*ratio)
random.shuffle(data)
data = data[:dsize]

with open('./data/wikisql_tok/train_tok.jsonl','w') as fout:
    for d in data:
        fout.write(json.dumps(d) + '\n')