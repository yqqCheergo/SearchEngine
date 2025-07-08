import numpy as np
with open("model_score.txt", 'r') as f:   # qids \t probs \t labels \t ... \t score
    data = f.readlines()

pred_score = []
label = []
q_idx = []
cur_idx = -1
q = ""

for i in range(1, len(data)):    # 第一行为header
    line = data[i].strip().split("\t")
    qid = line[0]
    score = line[-1]
    lab = line[2] 
    pred_score.append(float(score))
    label.append(int(lab))
    
    # 一次查询即一个qid，对应多个url的label和score
    if qid != q:
        q_idx.append([i])
        q = qid
        cur_idx += 1
    else:
        q_idx[cur_idx].append(i)

pos_cnt = 0
neg_cnt = 0
sep_cnt = 0

for urls in q_idx:
    for i in range(len(urls) - 1):
        for j in range(i + 1, len(urls)):
            x = urls[i] - 1
            y = urls[j] - 1

            try:
                res = (label[x] - label[y]) * (pred_score[x] - pred_score[y])
            except:
                print(x, y)

            if res > 0:
                pos_cnt += 1
            elif res < 0:
                neg_cnt += 1
            else:
                sep_cnt += 1

            # print(i, j, label[i], label[j], pred_score[i], pred_score[j], res)

print(pos_cnt)
print(neg_cnt)
print(sep_cnt)
print(pos_cnt / float(neg_cnt))    # 正逆序比PNR