from torch import nn, optim
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
from model_chatglm2 import model_chatglm2
from dataloader import data_compose, read_jsonl
from torch.utils.data import DataLoader
import os


def run_chatglm():
    train_data = ("/data/zhangxin/code/LLM/glm/tmp_nlptask/LLM/chatglm/data/暖气不热_nohistory.json")
    model_type = "/data/zhangxin/code/LLMModel/chatglm2-6b"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"  # 指定GPU编号
    device = torch.device("cuda")
    max_seq_length = 256
    batch_size = 2
    lr = 0.000001
    num_epochs = 3
    skip_overlength = False

    config = AutoConfig.from_pretrained(model_type, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
    # model_chatglm2 = AutoModel.from_pretrained(model_type, trust_remote_code=True)
    model = model_chatglm2(max_seq_length, batch_size)

    dataset = read_jsonl(tokenizer, config, train_data, max_seq_length, skip_overlength)
    train_dataset = data_compose(dataset, tokenizer)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    model.float()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(1, num_epochs + 1):
        for batch_idx, (input, label) in enumerate(train_loader):
            input, label = input.to(device), label.to(device)
            output = model(input)
            loss = criterion(output, label.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx % 10 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(input),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )


if __name__ == "__main__":
    run_chatglm()
