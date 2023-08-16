from torch import nn
import torch


class model_lstm(nn.Module):
    def __init__(
            self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, output_dim, fix_embedding=True,
    ):
        super().__init__()
        # 制作 embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 是否將 embedding fix住，如果fix_embedding为False，在训练过程中，embedding也会跟着被训练
        self.embedding.weight.requires_grad = False if fix_embedding else True

        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

        self.fc = nn.Linear(hidden_dim * 512, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        input = self.embedding(input)
        output, _ = self.lstm(input, None)
        # output 的 dimension (batch, seq_len, hidden_size)
        output = output.reshape(output.size()[0], -1)
        output = self.fc(output)
        return output
