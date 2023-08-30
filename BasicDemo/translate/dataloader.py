import re
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split


def read_txt(path):
    source = []
    target = []
    with open(path, "r", encoding="utf-8")as file:
        for line in file:
            line = re.sub(r"['?_.!+-=——,$%^，。！？、~@#￥%……&*《》<>「」{}【】()/]", '', line)
            line = line.strip().split("CCBY")[0]
            line = line.split("  ")
            line = [x for x in line if len(x) > 2]
            if len(line) == 2:
                source_line = [x for x in line[0].split(" ") if x != ' ']
                target_line = [x for x in line[1] if x != ' ']
                source.append(source_line)
                target.append(target_line)
    return source, target


def word_idx(data_list):
    vocab = set()
    word_to_idx = {"<sos>": 0, "<eos>": 1, "<pad>": 2}
    idx_to_word = {0: "<sos>", 1: "<eos>", 2: "<pad>"}
    for sent in data_list:
        for word in sent:
            vocab.add(word)
    for idx, word in enumerate(vocab):
        word_to_idx[word] = idx
        idx_to_word[idx] = word

    return word_to_idx, idx_to_word


def data_token(path, max_len):
    source, target = read_txt(path)
    source_word_to_idx, source_idx_to_word = word_idx(source)
    target_word_to_idx, target_idx_to_word = word_idx(target)
    source_vocab_len = len(source_word_to_idx)
    target_vocab_len = len(target_word_to_idx)

    train_source, test_source, train_target, test_target = train_test_split(
        source, target, test_size=0.3, random_state=0)

    def get_tensor(list_data, word_to_idx):
        sentences_idx = []
        for data in list_data:
            data.insert(0, "<sos>")
            if len(data) >= max_len:
                data = data[:max_len - 1]
                data.append("<eos>")
            else:
                data.append("<eos>")
                data.extend(["<pad>"] * (max_len - len(data)))
            sentence_idx = []
            for word in data:
                sentence_idx.append(word_to_idx[word])
            sentences_idx.append(sentence_idx)
        return sentences_idx

    train_source_tensor = torch.LongTensor(get_tensor(train_source, source_word_to_idx))
    train_target_tensor = torch.LongTensor(get_tensor(train_target, target_word_to_idx))

    test_source_tensor = torch.LongTensor(get_tensor(test_source, source_word_to_idx))
    test_target_tensor = torch.LongTensor(get_tensor(test_target, target_word_to_idx))

    train_dataset = TensorDataset(train_source_tensor, train_target_tensor)
    test_dataset = TensorDataset(test_source_tensor, test_target_tensor)

    return train_dataset, test_dataset, source_vocab_len, target_vocab_len


if __name__ == "__main__":
    max_len = 20
    path_txt = "/data/zhangxin/code/LLM/glm/tmp_nlptask/BasicDemo/translate/data/en_cn.txt"
    # source, target = read_txt(path_txt)
    # source_word_to_idx, source_idx_to_word = word_idx(source)
    # target_word_to_idx, target_idx_to_word = word_idx(target)
    # print(target_word_to_idx)
    train_dataset, test_dataset, source_vocab_len, target_vocab_len = data_token(path_txt, max_len)
    print(source_vocab_len,target_vocab_len)
    # for source, target in train_dataset:
    #     print(source)
