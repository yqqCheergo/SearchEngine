import math
from collections import defaultdict

# 文档集合（语料库）
corpus = [
    "The cat sat on the mat",
    "The dog ate my homework",
    "The cat and the dog are friends"
]
tokenized_corpus = [doc.lower().split() for doc in corpus]

# 参数设置
k = 1.5
b = 0.75


def compute_idf(tokenized_corpus):
    # 计算每个词的Document Frequency，即在语料库中的多少文档中出现过
    df = defaultdict(int)
    for doc in tokenized_corpus:
        seen_words = set(doc)
        for word in seen_words:
            df[word] += 1

    N = len(tokenized_corpus)  # 总文档数
    idf = {}
    for word, count in df.items():
        idf[word] = math.log(1 + (N - count + 0.5) / (count + 0.5))   # 公式中的第二项
    return idf


def compute_bm25(query, doc, tokenized_corpus):
    idf = compute_idf(tokenized_corpus)
    doc_len = [len(doc) for doc in tokenized_corpus]
    avg_dl = sum(doc_len) / len(doc_len)    # 语料库中文档的平均长度

    score = 0
    for word in query:
        if word not in doc:
            continue
        tf = doc.count(word)
        up = tf * (k + 1)
        down = tf + k * (1 - b + b * (len(doc) / avg_dl))
        score += idf.get(word, 0) * (up / down)
    return score

# 查询词
query = "cat and dog"
tokenized_query = query.lower().split()

# 计算所有文档的BM25分数
bm25_scores = []
for doc in tokenized_corpus:
    score = compute_bm25(tokenized_query, doc, tokenized_corpus)
    bm25_scores.append(score)
print("BM25分数:", bm25_scores)

# 找到BM25分数最高的文档
best_doc_idx = bm25_scores.index(max(bm25_scores))
print(f"最相关文档: '{corpus[best_doc_idx]}' (分数: {bm25_scores[best_doc_idx]:.2f})")