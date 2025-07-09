import math
from collections import defaultdict

# 文档集合（语料库）
corpus = [
    "the cat sat on the mat",
    "the dog ate my homework",
    "the cat and the dog are friends"
]

# 目标文档d和查询词q
document_d = "the cat on the mat"
query_q = "cat and dog"


# ------ 步骤1：计算语料库的IDF ------
def compute_idf(corpus):
    """计算逆文档频率（IDF）"""
    N = len(corpus)
    doc_count = defaultdict(int)
    
    for doc in corpus:
        seen_words = set(doc.split())
        for word in seen_words:
            doc_count[word] += 1    # 统计word在语料库中的多少文档中出现过，即df
    
    idf = {}
    for word, count in doc_count.items():
        idf[word] = math.log(N / (count + 1))   # +1避免除零
    return idf

idf = compute_idf(corpus)


# ------ 步骤2：计算目标文档d的TF ------
def compute_tf(text):
    """计算词频（TF）"""
    tf = defaultdict(int)
    words = text.split()
    for word in words:
        tf[word] += 1
    # 归一化
    total_words = len(words)   # 文档d中所有词的总数（包括重复词）
    for word in tf:
        tf[word] /= total_words
    return tf

tf_d = compute_tf(document_d)


# ------ 步骤3：计算目标文档d的TF—IDF ------
tfidf_d = {}
for word, tf_val in tf_d.items():
    tfidf_d[word] = tf_val * idf.get(word, 0)    # 未登录词IDF=0


# ------ 步骤4：计算查询词q和目标文档d的相关性 ------
query_words = query_q.split()
score = 0
for word in query_words:
    score += tfidf_d.get(word, 0)  # 累加q中分词在d中的TF-IDF值

print(f"查询词q和目标文档d的TF-IDF相关性分数: {score:.4f}")