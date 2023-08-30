import torch
import random
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

import colossalai
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin


class GRUModel(nn.Module):

    def __init__(self, input_num, hidden_num, output_num, vocab, embedding_dim):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_num
        self.embedding = nn.Embedding(vocab, embedding_dim)

        self.GRU_layer = nn.GRU(input_size=input_num, hidden_size=hidden_num, batch_first=True)
        self.output_linear = nn.Linear(hidden_num, output_num)
        self.hidden = None

    def forward(self, x):
        x = self.embedding(x)
        x, hidden = self.GRU_layer(x)
        hidden_out = hidden.squeeze(0)
        hidden_out = self.output_linear(hidden_out)
        return hidden_out


def create_data(DATA_NUM, SEQ_LEN, BATCH_SIZE):
    train_data = torch.rand(DATA_NUM, SEQ_LEN)
    label = torch.zeros(DATA_NUM)
    train_dataset = TensorDataset(train_data.long(), label.long())
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader


if __name__ == "__main__":

    # 参数设置
    EPOCH = 20
    BATCH_SIZE = 6
    DATA_NUM = 1000
    SEQ_LEN = 100
    INPUT_DIM = 200
    OUTPUT_DIM = 2
    HIDDEN_DIM = 100
    VOCAB = 3000
    EMBEDDING_DIM = 200

    # 加载数据
    train_loader = create_data(DATA_NUM, SEQ_LEN, BATCH_SIZE)

    # 加载模型
    gru = GRUModel(
        input_num=INPUT_DIM,
        hidden_num=HIDDEN_DIM,
        output_num=OUTPUT_DIM,
        vocab=VOCAB,
        embedding_dim=EMBEDDING_DIM)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(gru.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=1 / 2)

    colossalai.launch_from_torch(config={})
    coordinator = DistCoordinator()
    plugin = TorchDDPPlugin()
    booster = Booster(plugin=plugin)

    gru, optimizer, criterion, _, lr_scheduler = booster.boost(
        gru, optimizer, criterion=criterion, lr_scheduler=scheduler)

    for epoch in range(1, EPOCH + 1):
        gru.train()
        gru.cuda()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = gru(data)
            loss = criterion(output, target)
            booster.backward(loss, optimizer)
            optimizer.step()
            if batch_idx % 10 == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),))
        scheduler.step()
