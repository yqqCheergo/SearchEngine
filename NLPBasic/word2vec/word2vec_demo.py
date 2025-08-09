import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.FloatTensor


###### 文本预处理 ######
sentences = ["jack like dog", "jack like cat", "jack like animal",
             "dog cat animal", "banana apple cat dog like", "dog fish milk like",
             "dog cat animal like", "jack like apple", "apple like", "jack like banana",
             "apple banana jack movie book music like", "cat dog hate", "cat dog like"]

word_sequence = " ".join(sentences).split()   # ['jack', 'like', 'dog', 'jack', 'like', 'cat', ...]
vocab = list(set(word_sequence))
word2idx = {w: i for i, w in enumerate(vocab)}


###### 模型相关参数 ######
batch_size = 8
embedding_size = 2
C = 2   # window size
vocab_size = len(vocab)


###### 数据预处理 ######
skip_grams = []    # 中心词和背景词一一配对后的list
for idx in range(C, len(word_sequence) - C):    # 为了左右两边都有足够的背景词
    center = word2idx[word_sequence[idx]]
    context_idx = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
    context = [word2idx[word_sequence[i]] for i in context_idx]
    for w in context:
        skip_grams.append([center, w])

def make_data(skip_grams):
    input_data = []
    output_data = []
    for a, b in skip_grams:   # center, context
        # np.eye(vocab_size) 创建对角全1的矩阵，行=列=词数
        input_data.append(np.eye(vocab_size)[a])   # one-hot
        output_data.append(b)   # 标签不需要one-hot，只需要类别值
    return np.array(input_data, dtype=np.float32), np.array(output_data, dtype=np.int64)

input_data, output_data = make_data(skip_grams)
input_data = torch.from_numpy(input_data)    # numpy -> Tensor
output_data = torch.from_numpy(output_data).long()    # numpy -> LongTensor

dataset = Data.TensorDataset(input_data, output_data)
loader = Data.DataLoader(dataset, batch_size, shuffle=True)


###### 构建模型 ######
class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.W = nn.Parameter(torch.randn(vocab_size, embedding_size).type(dtype))
        self.V = nn.Parameter(torch.randn(embedding_size, vocab_size).type(dtype))

    def forward(self, X):
        # X: [batch_size, vocab_size]   one-hot
        # torch.mm() only for 2 dim matrix, but torch.matmul() can for any dim
        hidden_layer = torch.matmul(X, self.W)    # hidden_layer: [batch_size, embedding_size]
        output_layer = torch.matmul(hidden_layer, self.V)    # output_layer: [batch_size, vocab_size]
        return output_layer

model = Word2Vec().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


###### 训练 ######
for epoch in range(2000):
    for i, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)

        if (epoch + 1) % 1000 == 0:
            print(epoch + 1, i, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


###### 绘图 ######
# 由于这里每个词是用2维向量表示，所以可以在平面直角坐标系中标记出每个词，看看它们之间的距离
for i, label in enumerate(vocab):
    W, V = model.parameters()
    x, y = float(W[i][0]), float(W[i][1])    # W[i]是第i个单词的词向量，维度为(embedding_size, )  x和y为词向量的第0维和第1维
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')    # 用于在图表上添加文本标注，标注的位置由xy指定，文本内容由label提供

plt.show()