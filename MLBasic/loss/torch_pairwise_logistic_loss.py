'''
Reference: https://github.com/alibaba/EasyRec/blob/master/easy_rec/python/loss/pairwise_loss.py
PyTorch
    每个batch中有m个样本
    标签y1~ym，且假设有y1≥...≥ym
    模型打分p1~pm
    pairwise logistic loss的目标是最大化正序对数量。即对于所有满足yi>yj的样本对(i,j)，鼓励pi-pj尽量大
'''

import torch
import torch.nn.functional as F

def pairwise_logistic_loss(labels,
                           logits,
                           session_ids=None,     # 用于分组计算loss（如GAUC指标）
                           temperature=1.0,      # 用于缩放logits
                           hinge_margin=None,    # 正负样本对的logits最小差异
                           weights=1.0,          # 样本权重（标量或[batch_size]张量）
                           ohem_ratio=1.0,       # 困难样本比例
                           name=""):             # 损失名称

    loss_name = name if name else 'pairwise_logistic_loss'
    assert 0 < ohem_ratio <= 1.0, f"{loss_name} ohem_ratio must be in (0, 1]"

    # 1. 缩放logits
    if temperature != 1.0:
        logits = logits / temperature

    # 2. 计算所有样本对的分数差 si - sj。shape: [batch_size, batch_size]
    pairwise_logits = logits.unsqueeze(-1) - logits.unsqueeze(0)   # 前者[batch_size, 1]，后者[1, batch_size]

    # 3. 构建样本对掩码 (yi > yj)
    pairwise_mask = (labels.unsqueeze(-1) - labels.unsqueeze(0)) > 0   # [batch_size, batch_size]

    # 4. 可选：hinge margin约束
    if hinge_margin is not None:
        hinge_mask = pairwise_logits < hinge_margin
        pairwise_mask = pairwise_mask & hinge_mask

    # 5. 可选：仅计算同一session内的样本对
    if session_ids is not None:
        session_mask = (session_ids.unsqueeze(-1) == session_ids.unsqueeze(0))
        pairwise_mask = pairwise_mask & session_mask

    # 6. 提取有效对的分数差
    pairwise_logits = pairwise_logits[pairwise_mask]
    num_pair = torch.numel(pairwise_logits)

    # 7. 计算基础损失 log(1 + exp(-(si - sj)))
    losses = F.softplus(-pairwise_logits)   # 等价于 relu(-x) + log(1 + exp(-|x|)) 数值稳定实现

    # 8. 处理样本权重
    if isinstance(weights, torch.Tensor):
        weights = weights.unsqueeze(-1)  # [batch_size, 1]
        pairwise_weights = weights.repeat(1, weights.shape[0])  # [batch_size, batch_size]
        pairwise_weights = pairwise_weights[pairwise_mask]
    else:
        pairwise_weights = weights

# 9. 难例挖掘 (OHEM)
    if ohem_ratio == 1.0:
        return torch.sum(losses * pairwise_weights) / torch.max(
            torch.sum((pairwise_weights > 0).float()),
            torch.tensor(1.0)
        )
    else:
        weighted_losses = losses * pairwise_weights
        k = int(round(losses.numel() * ohem_ratio))
        topk_values, topk_indices = torch.topk(losses, k)
        topk_weighted_losses = weighted_losses[topk_indices]
        topk_weighted_losses = topk_weighted_losses[topk_weighted_losses > 0]
        return torch.mean(topk_weighted_losses)


########## 模拟数据调用函数 ##########
import numpy as np

batch_size = 4
np.random.seed(42)

labels = np.array([3.0, 1.0, 2.0, 0.0])  # 相关性分数
logits = np.random.normal(scale=0.5, size=batch_size)  # 缩小logits范围
session_ids = np.array([1, 1, 2, 2])      # 用户/会话分组
weights = np.array([1.0, 0.5, 1.5, 1.0])  # 样本权重

# 转换为Tensor
labels = torch.tensor(labels, dtype=torch.float32)
logits = torch.tensor(logits, dtype=torch.float32)
session_ids = torch.tensor(session_ids, dtype=torch.int32)
weights = torch.tensor(weights, dtype=torch.float32)

# 计算损失
loss1 = pairwise_logistic_loss(
    labels=labels,
    logits=logits,
    session_ids=session_ids,
    temperature=1.0,          # 不缩放logits
    hinge_margin=0.5,
    weights=weights,
    ohem_ratio=1.0,           # 先不使用OHEM
    name='pairwise_loss'
)

loss2 = pairwise_logistic_loss(
    labels=labels,
    logits=logits,
    session_ids=session_ids,
    temperature=0.9,          # 缩放logits
    hinge_margin=0.5,
    weights=weights,
    ohem_ratio=0.7,           # 使用70%最难样本
    name='pairwise_loss'
)

print("计算出的损失值: loss1 = ", loss1.numpy(), ", loss2 = ", loss2.numpy())      # 计算出的损失值: loss1 =  0.9752786 , loss2 =  1.4483577