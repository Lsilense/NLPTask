import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader
from config import args
from translate.dataloader import data_token
from model.seq2seq_lstm import Encoder, Decoder, Seq2Seq

# iterator -> train_iterator


def train(model, data_loader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    model.to(device)
    for i, (src, trg) in enumerate(data_loader):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    model.to(device)
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)  # turn off teacher forcing
            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


class run_model:
    def __init__(self):
        self.device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        train_dataset, test_dataset, source_vocab_len, target_vocab_len = data_token(args.path_txt, args.max_len)
        self.source_vocab_len = source_vocab_len
        self.target_vocab_len = target_vocab_len
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    def run_seq2seq_lstm(self):
        INPUT_DIM = self.source_vocab_len
        OUTPUT_DIM = self.target_vocab_len
        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        HID_DIM = 512
        N_LAYERS = 2
        ENC_DROPOUT = 0.5
        DEC_DROPOUT = 0.5

        encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
        decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
        model = Seq2Seq(encoder, decoder, self.device).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        N_EPOCHS = 10
        CLIP = 1
        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):
            train_loss = train(model, self.train_loader, optimizer, criterion, CLIP, self.device)
            valid_loss = evaluate(model, self.test_loader, criterion, self.device)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(
                    model.state_dict(),
                    '/data/zhangxin/code/LLM/glm/tmp_nlptask/BasicDemo/translate/output/tut1-model.pt')

            print(f'Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')


if __name__ == "__main__":
    run = run_model()
    run.run_seq2seq_lstm()
