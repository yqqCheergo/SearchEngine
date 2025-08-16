pairwise loss可以更好地逼近排序目标，它的重要意义在于让模型的训练目标和模型实际的任务之间尽量统一。对于一个排序任务，真实的目标是让正样本的预估分数比负样本的高，对应了AUC/GAUC这样的指标。

本目录下分别使用tensorflow 1.4.0、2.8.0以及pytorch实现了pairwise_logistic_loss，它在mini-batch内构建所有可能的正负样本对，或者构建在同一session内（当session_ids参数不为None时）的正负样本对。

该损失函数要求模型收敛后正样本的logit大于负样本的logit。

同时该损失函数还实现了困难样本挖掘的功能，可以通过ohem_ratio指定困难样本的比例，计算损失函数时只计算困难样本产生的loss，而忽略容易样本产生的loss，这样最后计算出来的loss值（reduce_mean之后的）会更大，也就是会产生更大的梯度，有利于模型的快速收敛。

$$
        \mathcal{L}(\{y\}, \{s\}) =
        \sum_i \sum_j I[y_i > y_j] \log(1 + \exp(-(s_i - s_j)))
$$

参考 https://github.com/alibaba/EasyRec/blob/master/easy_rec/python/loss/pairwise_loss.py