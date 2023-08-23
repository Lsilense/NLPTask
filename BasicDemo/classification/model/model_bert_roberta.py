import torch.nn as nn
from transformers import AutoModel


class model_bert_roberta(nn.Module):
    def __init__(self, model_name, hidden_size, output_dim, dropout):
        super(model_bert_roberta, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, device):
        input_ids, token_type_ids, attention_mask = (
            inputs[0].to(device),
            inputs[1].to(device),
            inputs[2].to(device),
        )
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # outputs.pooler_output: [bs, hidden_size]
        output = self.linear(self.dropout(outputs.pooler_output))

        return output
