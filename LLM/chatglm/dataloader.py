import json
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, AutoConfig


def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["instruction"]
    target = example["output"]
    prompt_ids = tokenizer.encode(
        prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target, max_length=max_seq_length, truncation=True, add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]

    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


# 调用preprocess 函数，将输入的特征转化为id的特征，将它们转化为我们想要的input_ids和长度格式
def read_jsonl(tokenizer, config, json_data, max_seq_length, skip_overlength=False):
    input_ids_list = []
    seqlen_list = []
    with open(json_data, mode="r", encoding="utf-8") as file:
        datas = json.load(file)
    for line in datas:
        feature = preprocess(tokenizer, config, line, max_seq_length)

        # 如果 skip_overlength 为 True，且特征的 input_ids 长度超过了 max_seq_length，则跳过该特征
        if skip_overlength and len(feature["input_ids"]) > max_seq_length:
            continue
        feature["input_ids"] = feature["input_ids"][:max_seq_length]

        input_ids_list.append(feature["input_ids"])
        seqlen_list.append(feature["seq_len"])

    return {"input_ids": input_ids_list, "seq_len": seqlen_list}


def data_compose(features: list, tokenizer):
    len_ids = [len(feature) for feature in features["input_ids"]]
    longest = max(len_ids)

    input_ids = []
    labels_list = []

    for ids_l, ids, seq_len in sorted(zip(len_ids, features["input_ids"], features["seq_len"]), key=lambda x: -x[0]):
        labels = ([-100] * (seq_len - 1) + ids[(seq_len - 1):] +
                  [-100] * (longest - ids_l))
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    train_dataset = TensorDataset(input_ids, labels)

    return train_dataset



if __name__ == "__main__":
    train_data = ("/data/zhangxin/code/LLM/glm/tmp_nlptask/LLM/chatglm/data/暖气不热_nohistory.json")
    model_type = "/data/zhangxin/code/LLMModel/chatglm2-6b"
    max_seq_length = 256
    skip_overlength = False

    config = AutoConfig.from_pretrained(model_type, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)

    dataset = read_jsonl(tokenizer, config, train_data, max_seq_length, skip_overlength)
    train_dataset = data_compose(dataset, tokenizer)
