import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model.lstm import model_lstm
from model.cnn import model_cnn
from dataloader import data_token, word_idx


def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    model.to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
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
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def run():
    # 参数配置
    device = torch.device("cuda:0")
    max_len = 512
    batch_size = 64
    num_epochs = 50
    csv_path = "/data/zhangxin/code/LLM/glm/tmp_nlptask/BasicDemo/data/ChnSentiCorp_htl_all.csv"

    # cnn 模型参数配置
    cnn_embedding_dim = 200
    cnn_n_filters = 100
    cnn_filter_sizes = [3, 4, 5]
    cnn_output_dim = 2
    cnn_dropout = 0.3

    # lstm 模型参数配置
    lstm_embedding_dim = 200
    lstm_hidden_dim = 200
    lstm_num_layers = 1
    lstm_dropout = 0.5
    lstm_output_dim = 2
    lstm_fix_embedding = True

    # 数据加载
    word_to_idx, _ = word_idx(csv_path)
    train_dataset, test_dataset = data_token(csv_path, max_len)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # cnn模型加载
    # model = model_cnn(
    #     vocab_size=len(word_to_idx),
    #     embedding_dim=cnn_embedding_dim,
    #     n_filters=cnn_n_filters,
    #     filter_sizes=cnn_filter_sizes,
    #     output_dim=cnn_output_dim,
    #     dropout=cnn_dropout,
    # )

    # lstm模型加载
    model = model_lstm(
        vocab_size=len(word_to_idx),
        embedding_dim=lstm_embedding_dim,
        hidden_dim=lstm_hidden_dim,
        num_layers=lstm_num_layers,
        dropout=lstm_dropout,
        output_dim=lstm_output_dim,
        fix_embedding=lstm_fix_embedding,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    # 开始训练
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, criterion)
        test(model, device, test_loader, criterion)


if __name__ == "__main__":
    run()
