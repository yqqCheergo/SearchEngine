# SearchEngine

本仓库旨在复现王树森老师《[搜索引擎技术](https://github.com/wangshusen/SearchEngine)》里涉及的算法/模型，以及一些NLP相关的算法/模型，工作闲暇之余抽空更新~

#### NLP基础

词向量模型：[Word2Vec](https://github.com/yqqCheergo/SearchEngine/tree/main/NLPBasic/word2vec)

#### 机器学习基础 (Chapter 2)

排序任务：[pairwise logistic loss](https://github.com/yqqCheergo/SearchEngine/tree/main/MLBasic/loss)

pairwise评价指标：[正逆序比PNR](https://github.com/yqqCheergo/SearchEngine/blob/main/MLBasic/cal_pnr.py)

#### 什么决定用户体验 (Chapter 3)

相关性：

- 词匹配分数：[TF-IDF](https://github.com/yqqCheergo/SearchEngine/blob/main/Rel/tf_idf.py)、[BM25](https://github.com/yqqCheergo/SearchEngine/blob/main/Rel/bm25.py)、[基于BM25从query、title、content中提取关键句](https://github.com/yqqCheergo/SearchEngine/blob/main/Rel/bm25_extract.py)
