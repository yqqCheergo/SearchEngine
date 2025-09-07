from collections import defaultdict
from sklearn.metrics import roc_auc_score
import numpy as np

def cal_group_auc(labels, preds, user_id_list):
    """
    Calculate group auc
        GAUC=\frac{\sum_{i=1}^{n}{\#impression_i×AUC_i}}{\sum_{i=1}^{n}{\#impression_i}}
    """

    if len(user_id_list) != len(labels):
        raise ValueError("impression id num should equal to the sample num," \
                         "impression id num is {0}".format(len(user_id_list)))

    # 按用户分组
    group_score = defaultdict(lambda: [])    # {user1: [score1, score2, ...], user2: [...]}
    group_truth = defaultdict(lambda: [])    # {user1: [label1, label2, ...], user2: [...]}
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)

    # 过滤掉单个用户全是正样本或负样本的情况
    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    # 计算GAUC
    impression_total = 0
    total_auc = 0

    for user_id in group_flag:
        if group_flag[user_id]:    # 只处理有正负样本混合的用户
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))    # list->ndarray
            # 这里实际有个前提假设，即数据集中只包含曝光过的样本，因此样本数就等于曝光量
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)   # 保留4位小数
    return group_auc

if __name__ == '__main__':
    # user1: 有正有负 | user2: 全正 | user3: 有正有负
    labels = [1, 0, 1, 1, 1, 1, 0, 1, 0, 0]
    preds = [0.8, 0.6, 0.7, 0.9, 0.9, 0.9, 0.3, 0.4, 0.2, 0.5]
    user_ids = ['user1', 'user1', 'user1', 'user2', 'user2', 'user2', 'user3', 'user3', 'user3', 'user3']

    gauc = cal_group_auc(labels, preds, user_ids)
    print(f"GAUC: {gauc}")

    # 计算过程
    auc_user1 = roc_auc_score([1, 0, 1], [0.8, 0.6, 0.7])   # 1.0
    # auc_user2 = roc_auc_score([1, 1, 1], [0.9, 0.9, 0.9])   # 全是正样本，跳过不计算
    auc_user3 = roc_auc_score([0, 1, 0, 0], [0.3, 0.4, 0.2, 0.5])   # 0.6667
    manual_gauc = (auc_user1 * 3 + auc_user3 * 4) / 7
    print(f"手动计算GAUC: {round(manual_gauc, 4)}")