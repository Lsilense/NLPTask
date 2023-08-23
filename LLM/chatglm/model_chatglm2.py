import torch
import torch.nn as nn
from transformers import AutoModel


class model_chatglm2(nn.Module):
    def __init__(self, seq_len, batch_size):
        super(model_chatglm2, self).__init__()
        self.chatglm2 = AutoModel.from_pretrained(
            "/data/zhangxin/code/LLMModel/chatglm2-6b", trust_remote_code=True
        )

        self.batch_size = batch_size
        self.fc = nn.Linear(seq_len * 65024, seq_len)

    def forward(self, inputs):
        outputs = self.chatglm2(inputs)["logits"]
        outputs = outputs.view(self.batch_size, -1).float()
        outputs = self.fc(outputs)

        return outputs
