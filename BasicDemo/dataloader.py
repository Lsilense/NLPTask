from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from torch.utils.data import TensorDataset


def read_csv(csv_path):
    datas = pd.read_csv(csv_path)
    train_data_list = []
    label_list = []
    for review, label in zip(datas["review"], datas["label"]):
        train_data_list.append(review)
        label_list.append(label)
    return train_data_list, label_list


def word_idx(csv_path):
    word_to_idx = {}
    idx_to_word = {}
    train_data_list, label_list = read_csv(csv_path)
    vocab = set()
    for data in train_data_list:
        for word in data:
            vocab.add(word)
    for idx, word in enumerate(vocab):
        word_to_idx[word] = idx
        idx_to_word[idx] = word
    idx_to_word[len(vocab)] = "<PAD>"
    word_to_idx["<PAD>"] = len(vocab)
    return word_to_idx, idx_to_word


def data_token(csv_path, max_len):
    train_data_list, label_list = read_csv(csv_path)
    word_to_idx, idx_to_word = word_idx(csv_path)
    sentences_idx = []
    for data in train_data_list:
        data = [x for x in data]
        if len(data) >= max_len:
            data = data[:max_len]
        else:
            data.extend(["<PAD>"] * (max_len - len(data)))
        sentence_idx = []
        for word in data:
            sentence_idx.append(word_to_idx[word])
        sentences_idx.append(sentence_idx)

    train_sentences, test_sentences, train_label, test_label = train_test_split(
        sentences_idx, label_list, test_size=0.3, random_state=0
    )
    train_sentences_tensor = torch.LongTensor(train_sentences)
    train_label_tensor = torch.LongTensor(train_label)
    test_sentences_tensor = torch.LongTensor(test_sentences)
    test_label_tensor = torch.LongTensor(test_label)
    train_dataset = TensorDataset(train_sentences_tensor, train_label_tensor)
    test_dataset = TensorDataset(test_sentences_tensor, test_label_tensor)

    return train_dataset, test_dataset


if __name__ == "__main__":
    max_len = 512
    csv_path = "/data/zhangxin/code/LLM/glm/tmp_nlptask/BasicDemo/data/ChnSentiCorp_htl_all.csv"
    train_dataset, test_dataset = data_token(csv_path, max_len)
    print(train_dataset, test_dataset)
