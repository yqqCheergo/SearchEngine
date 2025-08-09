import copy
import random
from collections import Counter
import numpy as np
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

C = 3   # 选取中心词左右多少个单词作为背景词，即window size
K = 15   # 负采样近似训练。每个正样本对应K个负样本
MAX_VOCAB_SIZE = 10000   # 词典大小。这里只选语料库中出现次数最多的9999个词，还有一个词是<UNK>来表示所有的其他词
EMBEDDING_SIZE = 100   # 每个词的词向量维度
epochs = 2
batch_size = 32
lr = 1e-3


###### 读取文本数据并处理 ######
# with open('text8.train.txt') as f:    # text8语料下载地址 http://mattmahoney.net/dc/text8.zip
#     text = f.read()
# test text
text = 'anarchism originated as a term of abuse first used against early working class radicals including the diggers of the english revolution and the sans culottes of the french revolution whilst the term is still used in a pejorative way to describe any act that used two america computer'
text = text.lower().split()   # 单词列表

vocab_dict = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))     # key是单词，value是频次，按频次从高到低排序
vocab_dict['<UNK>'] = len(text) - np.sum(list(vocab_dict.values()))    # 把不常用的单词都编码为<UNK>

# 词典中的每个词 与 0~词典大小-1的整数 一一对应
word2idx = {word: i for i, word in enumerate(vocab_dict.keys())}
idx2word = {i: word for i, word in enumerate(vocab_dict.keys())}

word_counts = np.array([count for count in vocab_dict.values()], dtype=np.float32)
word_counts = word_counts ** 0.75
word_freqs = word_counts / sum(word_counts)


###### 实现DataLoader ######
class WordEmbeddingDataset(Data.Dataset):
    def __init__(self, text, word2idx, word_freqs):
        super(WordEmbeddingDataset, self).__init__()    # 通过父类初始化模型，重写两个方法
        self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text]    # 把单词数字化表示。如果不在词典中，返回<UNK>对应的数字
        self.text_encoded = torch.LongTensor(self.text_encoded)    # nn.Embedding() 需要传入LongTensor类型
        self.word2idx = word2idx
        self.word_freqs = torch.Tensor(word_freqs)

    def __len__(self):
        return len(self.text_encoded)    # 返回所有单词的总数

    def __getitem__(self, idx):
        '''
        返回以下数据用于训练：
            中心词
            中心词附近的背景词 positive word
            随机采样 K个单词作为 negative word
        '''
        center_words = self.text_encoded[idx]   # 中心词 (数字化表示)
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))   # 中心词左右各C个词的索引
        pos_indices = [i if 0 <= i < len(self.text_encoded) else self.word2idx['<UNK>'] for i in pos_indices]    # 索引越界(即中心词左右不足C个词) 一律视为<UNK>
        pos_words = self.text_encoded[pos_indices]   # 背景词 & 正样本
        # 以下3行代码为保证采样的neg_words中不包含中心词和背景词
        select_weight = copy.deepcopy(self.word_freqs)
        select_weight[pos_words], select_weight[center_words] = 0, 0
        neg_words = torch.multinomial(select_weight, K * pos_words.shape[0], True)    # 根据词频分布，采样 K * len(pos_words) 个负样本 (每采样一个正确的单词，就采样K个错误的单词)。True表示允许重复采样
        return center_words, pos_words, neg_words

dataset = WordEmbeddingDataset(text, word2idx, word_freqs)
dataloader = Data.DataLoader(dataset, batch_size, shuffle=True)
# print(next(iter(dataset)))    # 打印一个样本看看


###### 定义Pytorch模型 ######
# 参考 https://wmathor.com/index.php/archives/1430/
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        # 任一个词都既有可能作为中心词出现，也有可能作为背景词出现，所以每个词需要用两个向量去表示
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)     # 每个词作为中心词的权重
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)    # 每个词作为背景词的权重

    def forward(self, input_labels, pos_labels, neg_labels):
        '''
            input_labels: center_words, [batch_size]
            pos_labels: positive_words 即实际出现的上下文词, [batch_size, (window_size * 2)]
            neg_labels: negative_words 即随机采样的非上下文词, [batch_size, (window_size * 2 * K)]

            return: loss, [batch_size]
        '''
        input_embedding = self.in_embed(input_labels)    # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)       # [batch_size, (window_size * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels)       # [batch_size, (window_size * 2 * K), embed_size]

        input_embedding = torch.unsqueeze(input_embedding, 2)    # [batch_size, embed_size, 1]

        pos_dot = torch.bmm(pos_embedding, input_embedding)    # [batch_size, (window_size * 2), 1]
        pos_dot = torch.squeeze(pos_dot, 2)   # [batch_size, (window_size * 2)]

        neg_dot = torch.bmm(neg_embedding, input_embedding)    # [batch_size, (window_size * 2 * K), 1]
        neg_dot = torch.squeeze(neg_dot, 2)   # [batch_size, (window_size * 2 * K)]

        log_pos = F.logsigmoid(pos_dot).sum(1)    # [batch_size, 1]
        log_neg = F.logsigmoid(-neg_dot).sum(1)    # 1-sigmoid(x) = sigmoid(-x)  负例算交叉熵损失时为 log(1-y')，其中 y' = sigmoid(z)
        loss = log_pos + log_neg
        return -loss

    def input_embedding(self):
        return self.in_embed.weight.detach().numpy()    # Word2Vec paper推荐用中心词向量表示一个词

model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


###### 训练模型 ######
for epoch in range(epochs):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()

        loss = model(input_labels, pos_labels, neg_labels).mean()

        optimizer.zero_grad()   # 清零梯度
        loss.backward()   # 反向传播计算梯度
        optimizer.step()   # 更新参数

        if i % 100 == 0:
            print('epoch: ', epoch, 'iteration: ', i, loss.item())

embedding_weights = model.input_embedding()     # [vocab_size, embed_size]
torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))


###### 词向量应用：找出与某个词相近的一些词 ######
def find_nearest(word):
    index = word2idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array(cosine(emb, embedding) for emb in embedding_weights)    # 计算两个向量之间的余弦距离 (1-余弦相似度)，越小越相似
    return [idx2word[i] for i in cos_dis.argsort()[:10]]    # .argsort()返回升序排序后的索引

for word in ["two", "america", "computer"]:
    print(word, find_nearest(word))