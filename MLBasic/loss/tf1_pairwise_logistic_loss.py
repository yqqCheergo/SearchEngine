'''
Reference: https://github.com/alibaba/EasyRec/blob/master/easy_rec/python/loss/pairwise_loss.py
tf 1.4.0
    每个batch中有m个样本
    标签y1~ym，且假设有y1≥...≥ym
    模型打分p1~pm
    pairwise logistic loss的目标是最大化正序对数量。即对于所有满足yi>yj的样本对(i,j)，鼓励pi-pj尽量大
'''

import tensorflow as tf
from tensorflow.python.ops.losses.losses_impl import compute_weighted_loss
import logging

def pairwise_logistic_loss(labels,
                           logits,
                           session_ids=None,        # 用于分组计算loss（如GAUC指标）
                           temperature=1.0,         # 用于缩放logits
                           hinge_margin=None,       # 正负样本对的logits差异至少达到该值
                           weights=1.0,             # 样本权重（标量，或[batch_size]的张量）
                           ohem_ratio=1.0,          # 指定困难样本的比例
                           name=''):                # 损失名称，用于日志和监控
    """Computes pairwise logistic loss between `labels` and `logits`.

    Definition:
        $$
        \mathcal{L}(\{y\}, \{s\}) =
        \sum_i \sum_j I[y_i > y_j] \log(1 + \exp(-(s_i - s_j)))
        $$

    Args:
        labels: A `Tensor` of the same shape as `logits` representing graded relevance.
        logits: A `Tensor` with shape [batch_size].
        session_ids: a `Tensor` with shape [batch_size]. Session ids of each sample, used to max GAUC metric. e.g. user_id
        temperature: (Optional) The temperature to use for scaling the logits.
        hinge_margin: the margin between positive and negative logits
        weights: A scalar, a `Tensor` with shape [batch_size] for each sample
        ohem_ratio: the percent of hard examples to be mined
        name: the name of loss
    """

    loss_name = name if name else 'pairwise_logistic_loss'
    assert 0 < ohem_ratio <= 1.0, loss_name + ' ohem_ratio must be in (0, 1]'
    logging.info('[{}] hinge margin: {}, ohem_ratio: {}, temperature: {}'.format(loss_name, hinge_margin, ohem_ratio, temperature))

    if temperature != 1.0:
        logits /= temperature   # 调整logits的数值范围

    # 利用广播机制，生成所有样本对(i,j)的logits差。shape: [batch_size, batch_size]
    pairwise_logits = tf.subtract(tf.expand_dims(logits, -1), tf.expand_dims(logits, 0))   # 前者[batch_size, 1]，后者[1, batch_size]
    # 标记所有满足yi>yj的样本对(i,j)，返回true or false
    pairwise_mask = tf.greater(tf.expand_dims(labels, -1) - tf.expand_dims(labels, 0), 0)

    if hinge_margin is not None:
        hinge_mask = tf.less(pairwise_logits, hinge_margin)    # logits < margin 返回true，此时才需计算loss
        pairwise_mask = tf.logical_and(pairwise_mask, hinge_mask)

    if session_ids is not None:
        logging.info('[%s] use session ids' % loss_name)
        group_equal = tf.equal(tf.expand_dims(session_ids, -1), tf.expand_dims(session_ids, 0))   # 只计算同一session内的样本对
        pairwise_mask = tf.logical_and(pairwise_mask, group_equal)

    pairwise_logits = tf.boolean_mask(pairwise_logits, pairwise_mask)   # 保留true位置对应的元素
    num_pair = tf.size(pairwise_logits)
    tf.summary.scalar('loss/%s_num_of_pairs' % loss_name, num_pair)   # 写入标量结果到TensorBoard

    # 等价于 log(1 + exp(-pairwise_logits))，通过 relu 和 log1p 实现数值稳定性
    losses = tf.nn.relu(-pairwise_logits) + tf.log1p(tf.exp(-tf.abs(pairwise_logits)))

    # 加权损失
    if tf.is_numeric_tensor(weights):   # 张量权重
        logging.info('[%s] use sample weight' % loss_name)
        weights = tf.expand_dims(tf.cast(weights, tf.float32), -1)   # weights必须是一个二维张量 [batch_size, 1]
        batch_size = weights.get_shape()[0].value
        pairwise_weights = tf.tile(weights, tf.stack([1, batch_size]))   # [batch_size, batch_size]
        pairwise_weights = tf.boolean_mask(pairwise_weights, pairwise_mask)   # 保留true位置对应的元素
    else:   # 标量权重
        pairwise_weights = weights

    # 难例挖掘。=1为使用所有样本，即不进行难例挖掘
    if ohem_ratio == 1.0:
        return compute_weighted_loss(losses, pairwise_weights)   # 默认是 Reduction.SUM_BY_NONZERO_WEIGHTS，计算公式为 sum(losses * weights) / max(1, count(weights > 0))

    # Reduction.NONE  不归约，保持每个样本对的独立损失
    losses = compute_weighted_loss(losses, pairwise_weights, reduction=tf.losses.Reduction.NONE)   # losses * weights
    # 进行难例挖掘时，需要保留的样本对数量。如 ohem_ratio=0.7 表示使用 70% 最难样本
    k = tf.to_float(tf.size(losses)) * tf.convert_to_tensor(ohem_ratio)
    k = tf.to_int32(tf.rint(k))   # 四舍五入到最接近的整数
    topk = tf.nn.top_k(losses, k)    # 选取损失最大的前k个样本对
    losses = tf.boolean_mask(topk.values, topk.values > 0)    # 确保只保留损失值为正（即模型预测错误）的样本进行梯度更新
    return tf.reduce_mean(losses)    # 默认计算所有元素的均值


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


with tf.Session() as sess:
    loss1 = sess.run(loss1)
    loss2 = sess.run(loss2)
    print("计算出的损失值: loss1 = ", loss1, ", loss2 = ", loss2)      # 计算出的损失值: loss1 =  0.9752786 , loss2 =  1.4483577