import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model.model_bert_roberta import model_bert_roberta
from model.lstm import model_lstm
from model.cnn import model_cnn
from classification.dataloader import bert_token, data_token, word_idx
from config import args
# from torch.utils.tensorboard import SummaryWriter
import numpy as np


def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    model.to(device)
    for batch_idx, data in enumerate(train_loader):
        target = data[-1].to(device)
        optimizer.zero_grad()
        output = model(data, device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


# Testing function
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            target = data[-1].to(device)
            output = model(data, device)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n".format(
          test_loss,
          correct,
          len(test_loader.dataset),
          100.0 * correct / len(test_loader.dataset)
          ))


class run_model:
    def __init__(self):
        self.device = torch.device(args.cuda)
        self.vocab_size = len(word_idx(args.csv_path)[0])

        # 加载分类数据集
        train_dataset, test_dataset = data_token(args.csv_path, args.max_len)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

        # 加载分类数据集bert编码
        bert_train_dataset, bert_test_dataset = bert_token(args.csv_path, args.max_len, args.bert_roberta_name)
        self.bert_train_loader = DataLoader(dataset=bert_train_dataset, batch_size=args.batch_size, shuffle=True)
        self.bert_test_loader = DataLoader(dataset=bert_test_dataset, batch_size=args.batch_size, shuffle=False)

    def run_cnn(self):
        model = model_cnn(
            vocab_size=self.vocab_size,
            embedding_dim=args.cnn_embedding_dim,
            n_filters=args.cnn_n_filters,
            filter_sizes=args.cnn_filter_sizes,
            output_dim=args.cnn_output_dim,
            dropout=args.cnn_dropout,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.cnn_lr)

        for epoch in range(1, args.num_epochs + 1):
            train(model, self.device, self.train_loader, optimizer, epoch, criterion)
            test(model, self.device, self.test_loader, criterion)

    def run_lstm(self):
        model = model_lstm(
            max_len=args.max_len,
            vocab_size=self.vocab_size,
            embedding_dim=args.lstm_embedding_dim,
            hidden_dim=args.lstm_hidden_dim,
            num_layers=args.lstm_num_layers,
            dropout=args.lstm_dropout,
            output_dim=args.lstm_output_dim,
            fix_embedding=args.lstm_fix_embedding,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lstm_lr)

        for epoch in range(1, args.num_epochs + 1):
            train(model, self.device, self.train_loader, optimizer, epoch, criterion)
            test(model, self.device, self.test_loader, criterion)

    def run_bert_roberta(self):
        model = model_bert_roberta(
            model_name=args.bert_roberta_name,
            hidden_size=args.bert_hidden_size,
            output_dim=args.bert_output_dim,
            dropout=args.bert_dropout,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.bert_lr)

        for epoch in range(1, args.num_epochs + 1):
            train(
                model, self.device, self.bert_train_loader, optimizer, epoch, criterion
            )
            test(model, self.device, self.bert_test_loader, criterion)


if __name__ == "__main__":

    # tensorboard --logdir='/data/zhangxin/code/LLM/glm/tmp_nlptask/BasicDemo/data/run_log' --port='6007'
    # writer = SummaryWriter("/data/zhangxin/code/LLM/glm/tmp_nlptask/BasicDemo/data/run_log")
    # for x in range(1, 101):
    #     writer.add_scalar('y = 2x', x, 2 * x)  # 这里反了，正常情况应该是writer.add_scalar('y = 2x', 2 * x, x)
    # writer.close()

    run = run_model()
    print("------cnn模型训练-----")
    run.run_cnn()
    # print("------lstm模型训练-----")
    # run.run_lstm()
    # print("--{}模型训练--".format(args.bert_roberta_name))
    # run.run_bert_roberta()
