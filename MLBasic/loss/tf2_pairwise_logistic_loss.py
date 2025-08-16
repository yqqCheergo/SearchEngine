'''
Reference: https://github.com/alibaba/EasyRec/blob/master/easy_rec/python/loss/pairwise_loss.py
tf 2.8.0
    每个batch中有m个样本
    标签y1~ym，且假设有y1≥...≥ym
    模型打分p1~pm
    pairwise logistic loss的目标是最大化正序对数量。即对于所有满足yi>yj的样本对(i,j)，鼓励pi-pj尽量大
'''

import tensorflow as tf
import logging

def pairwise_logistic_loss(labels,
                           logits,
                           session_ids=None,     # 用于分组计算loss（如GAUC指标）
                           temperature=1.0,      # 用于缩放logits
                           hinge_margin=None,    # 正负样本对的logits最小差异
                           weights=1.0,          # 样本权重（标量或[batch_size]张量）
                           ohem_ratio=1.0,       # 困难样本比例
                           name=''):             # 损失名称

    loss_name = name if name else 'pairwise_logistic_loss'
    assert 0 < ohem_ratio <= 1.0, loss_name + ' ohem_ratio must be in (0, 1]'
    logging.info(f"[{loss_name}] hinge_margin={hinge_margin}, ohem_ratio={ohem_ratio}, temperature={temperature}")

    # 1. 缩放logits
    logits = logits / temperature

    # 2. 计算所有样本对的logits差。shape: [batch_size, batch_size]
    pairwise_logits = tf.expand_dims(logits, -1) - tf.expand_dims(logits, 0)   # 前者[batch_size, 1]，后者[1, batch_size]

    # 3. 构建样本对掩码 (yi > yj)
    pairwise_mask = tf.greater(
        tf.expand_dims(labels, -1) - tf.expand_dims(labels, 0),
        0
    )

    # 4. 可选：hinge margin约束
    if hinge_margin is not None:
        hinge_mask = tf.less(pairwise_logits, hinge_margin)    # logits < margin 返回true，此时才需计算loss
        pairwise_mask = tf.logical_and(pairwise_mask, hinge_mask)

    # 5. 可选：按session_id分组
    if session_ids is not None:
        logging.info('[%s] use session ids' % loss_name)
        group_mask = tf.equal(
            tf.expand_dims(session_ids, -1),
            tf.expand_dims(session_ids, 0)
        )
        pairwise_mask = tf.logical_and(pairwise_mask, group_mask)

    # 6. 提取有效样本对
    pairwise_logits = tf.boolean_mask(pairwise_logits, pairwise_mask)
    num_pair = tf.size(pairwise_logits)
    tf.summary.scalar(f'loss/{loss_name}_num_of_pairs', num_pair)

    # 7. 计算基础损失 (数值稳定实现)
    losses = tf.nn.relu(-pairwise_logits) + tf.math.log1p(tf.exp(-tf.abs(pairwise_logits)))

    # 8. 加权损失
    if isinstance(weights, tf.Tensor):
        logging.info('[%s] use sample weight' % loss_name)
        weights = tf.expand_dims(tf.cast(weights, tf.float32), -1)   # [batch_size, 1]
        pairwise_weights = tf.tile(weights, [1, tf.shape(weights)[0]])    # 生成样本对权重矩阵 [batch_size, batch_size]
        pairwise_weights = tf.boolean_mask(pairwise_weights, pairwise_mask)
    else:
        pairwise_weights = weights

    # 9. 难例挖掘 (OHEM)
    if ohem_ratio == 1.0:
        # 加权平均损失 (替代compute_weighted_loss), 计算公式: sum(losses * weights) / max(1, count(weights > 0))
        return tf.reduce_sum(losses * pairwise_weights) / tf.maximum(
            tf.reduce_sum(tf.cast(pairwise_weights > 0, tf.float32)),
            1.0
        )
    else:
        # 保持每个样本对的独立损失
        weighted_losses = losses * pairwise_weights
        k = tf.cast(tf.size(losses), tf.float32) * ohem_ratio   # 使用ohem_ratio比例的最难样本
        k = tf.cast(tf.round(k), tf.int32)   # 四舍五入到最接近的整数
        topk = tf.nn.top_k(losses, k=k)   # 选取损失最大的前k个样本对
        topk_weighted_losses = tf.gather(weighted_losses, topk.indices)
        topk_weighted_losses = tf.boolean_mask(topk_weighted_losses, topk_weighted_losses > 0)   # 只保留损失值为正（即模型预测错误）的样本进行梯度更新
        return tf.reduce_mean(topk_weighted_losses)   # 计算均值


########## 模拟数据调用函数 ##########
import numpy as np

batch_size = 4
np.random.seed(42)

labels = np.array([3.0, 1.0, 2.0, 0.0])  # 相关性分数
logits = np.random.normal(scale=0.5, size=batch_size)  # 缩小logits范围
session_ids = np.array([1, 1, 2, 2])      # 用户/会话分组
weights = np.array([1.0, 0.5, 1.5, 1.0])  # 样本权重

# 转换为Tensor
labels = tf.constant(labels, dtype=tf.float32)
logits = tf.constant(logits, dtype=tf.float32)
session_ids = tf.constant(session_ids, dtype=tf.int32)
weights = tf.constant(weights, dtype=tf.float32)

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