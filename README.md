# SearchEngine

本仓库旨在复现王树森老师《[搜索引擎技术](https://github.com/wangshusen/SearchEngine)》里涉及的算法/模型，以及一些NLP相关的算法/模型，工作闲暇之余抽空更新~

#### NLP基础

词向量模型：[Word2Vec](https://github.com/yqqCheergo/SearchEngine/tree/main/NLPBasic/word2vec)

#### 机器学习基础 (Chapter 2)

二分类任务：[交叉熵损失Cross Entropy](https://github.com/yqqCheergo/SearchEngine/blob/main/MLBasic/loss/cross_entropy_loss.py)

排序任务：[pairwise logistic loss(Tensorflow1.4.0)](https://github.com/yqqCheergo/SearchEngine/blob/main/MLBasic/loss/tf1_pairwise_logistic_loss.py) & [pairwise logistic loss(Tensorflow2.8.0)](https://github.com/yqqCheergo/SearchEngine/blob/main/MLBasic/loss/tf2_pairwise_logistic_loss.py) & [pairwise logistic loss(PyTorch)](https://github.com/yqqCheergo/SearchEngine/blob/main/MLBasic/loss/torch_pairwise_logistic_loss.py) 以及[对应说明](https://github.com/yqqCheergo/SearchEngine/blob/main/MLBasic/loss/loss.md)

pointwise评价指标：[AUC(Python)](https://github.com/yqqCheergo/SearchEngine/blob/main/MLBasic/auc.py) & [AUC(C++)](https://github.com/yqqCheergo/SearchEngine/blob/main/MLBasic/cal_auc.cpp) 以及拓展的[GAUC(Python)](https://github.com/yqqCheergo/SearchEngine/blob/main/MLBasic/gauc.py)

pairwise评价指标：[正逆序比PNR](https://github.com/yqqCheergo/SearchEngine/blob/main/MLBasic/cal_pnr.py)

#### 什么决定用户体验 (Chapter 3)

相关性：

- 词匹配分数：[TF-IDF](https://github.com/yqqCheergo/SearchEngine/blob/main/Rel/tf_idf.py)、[BM25](https://github.com/yqqCheergo/SearchEngine/blob/main/Rel/bm25.py)、[基于BM25从query、title、content中提取关键句](https://github.com/yqqCheergo/SearchEngine/blob/main/Rel/bm25_extract.py)
