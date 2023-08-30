from torch import nn, optim
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
from chatglm2_model_detail import model_chatglm2
from dataloader import data_compose, read_jsonl
from torch.utils.data import DataLoader
import os

import colossalai
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin


def run_chatglm():
    train_data = ("/data/zhangxin/code/LLM/glm/tmp_nlptask/LLM/chatglm/data/暖气不热_nohistory.json")
    model_type = "/data/zhangxin/code/LLMModel/chatglm2-6b"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"  # 指定GPU编号
    device = torch.device("cuda")
    max_seq_length = 64
    batch_size = 1
    lr = 0.01
    num_epochs = 3
    skip_overlength = False

    config = AutoConfig.from_pretrained(model_type, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
    model = model_chatglm2(max_seq_length, batch_size)

    for name, value in model.named_parameters():
        if "fc" not in name:
            value.requires_grad = False

    dataset = read_jsonl(tokenizer, config, train_data, max_seq_length, skip_overlength)
    train_dataset = data_compose(dataset, tokenizer)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model.float()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=1 / 2)

    colossalai.launch_from_torch(config={})
    coordinator = DistCoordinator()
    plugin = TorchDDPPlugin()
    booster = Booster(plugin=plugin)

    model, optimizer, criterion, _, lr_scheduler = booster.boost(
        model, optimizer, criterion=criterion, lr_scheduler=scheduler)

    model.to(device)
    for epoch in range(1, num_epochs + 1):
        for batch_idx, (input, label) in enumerate(train_loader):
            input, label = input.to(device), label.to(device)
            output = model(input)
            loss = criterion(output, label.float())
            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx % 10 == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(input),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                ))
        scheduler.step()


if __name__ == "__main__":
    run_chatglm()
