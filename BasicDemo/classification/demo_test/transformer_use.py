import math
import random

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class CopyTaskModel(nn.Module):

    def __init__(self, d_model=1024):
        super(CopyTaskModel, self).__init__()

        # 定义词向量，词典数为3000。
        self.embedding = nn.Embedding(num_embeddings=3000, embedding_dim=1024)
        # 定义Transformer。超参是我拍脑袋想的
        self.transformer = nn.Transformer(
            d_model=1024,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            batch_first=True)

        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout=0)

        # 定义最后的线性层，这里并没有用Softmax，因为没必要。
        # 因为后面的CrossEntropyLoss中自带了
        self.predictor = nn.Linear(1024, 3000)

    def forward(self, src, tgt):
        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1])
        src_key_padding_mask = CopyTaskModel.get_key_padding_mask(src)
        tgt_key_padding_mask = CopyTaskModel.get_key_padding_mask(tgt)

        # 对src和tgt进行编码
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # 将准备好的数据送给transformer
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)

        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 2] = -torch.inf
        return key_padding_mask


def generate_random_batch(batch_size, max_length):
    src = []
    for i in range(batch_size):
        # 随机生成句子长度
        random_len = random.randint(1, max_length - 2)
        # 随机生成句子词汇，并在开头和结尾增加<bos>和<eos>
        random_nums = [0] + [random.randint(3, 3000) for _ in range(random_len)] + [1]
        # 如果句子长度不足max_length，进行填充
        random_nums = random_nums + [2] * (max_length - random_len - 2)
        src.append(random_nums)
    src = torch.LongTensor(src)
    # tgt不要最后一个token
    tgt = src[:, :-1]
    # tgt_y不要第一个的token
    tgt_y = src[:, 1:]
    # 计算tgt_y，即要预测的有效token的数量
    n_tokens = (tgt_y != 2).sum()

    # 这里的n_tokens指的是我们要预测的tgt_y中有多少有效的token，后面计算loss要用
    return src, tgt, tgt_y, n_tokens


model = CopyTaskModel()
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

total_loss = 0
max_length = 100

for step in range(2000):
    # 生成数据
    src, tgt, tgt_y, n_tokens = generate_random_batch(batch_size=2, max_length=max_length)

    # 清空梯度
    optimizer.zero_grad()
    # 进行transformer的计算
    out = model(src, tgt)
    # 将结果送给最后的线性层进行预测
    out = model.predictor(out)
    """
    计算损失。由于训练时我们的是对所有的输出都进行预测，所以需要对out进行reshape一下。
            我们的out的Shape为(batch_size, 词数, 词典大小)，view之后变为：
            (batch_size*词数, 词典大小)。
            而在这些预测结果中，我们只需要对非<pad>部分进行，所以需要进行正则化。也就是
            除以n_tokens。
    """
    loss = criteria(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1)) / n_tokens
    # 计算梯度
    loss.backward()
    # 更新参数
    optimizer.step()

    total_loss += loss

    # 每40次打印一下loss
    if step != 0 and step % 40 == 0:
        print("Step {}, total_loss: {}".format(step, total_loss))
        total_loss = 0
