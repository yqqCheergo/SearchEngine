import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

### 公式法计算AUC ###
def calAUC(prob, label):
    data = list(zip(prob, label))   # 将预测概率和真实标签拼在一起，形成二元组
    rank = [label for prob, label in sorted(data, key=lambda x: x[0])]   # 按prob升序排列
    rank_list = [i + 1 for i in range(len(rank)) if rank[i] == 1]   # 正样本位置序列

    pos_num, neg_num = 0, 0
    for i in range(len(label)):
        if label[i] == 1:
            pos_num += 1
        else:
            neg_num += 1

    return (sum(rank_list) - pos_num * (pos_num + 1) / 2) / (pos_num * neg_num)

if __name__ == '__main__':
    y = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0])
    pred = np.array([0.9, 0.4, 0.85, 0.3, 0.1, 0.35, 0.6, 0.65, 0.32, 0.8, 0.7, 0.2])
    auc_value = calAUC(pred, y)
    print(auc_value)

    ### 直接调库计算AUC ###
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    print(auc(fpr, tpr))
    print(roc_auc_score(y, pred))
    plt.plot(fpr, tpr)    # roc曲线横轴为FPR, 纵轴为TPR
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC')
    plt.show()