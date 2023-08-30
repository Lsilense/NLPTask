from torch import nn, optim
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
from chatglm2_model_detail import model_chatglm2
from dataloader import data_compose, read_jsonl
from torch.utils.data import DataLoader
import os
from transformers import TrainingArguments
from transformers import Trainer
import datasets


model_type = "/data/zhangxin/code/LLMModel/chatglm2-6b"
tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    # 重写 save_model 方法，保存模型
    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)

    input_ids = []
    labels_list = []

    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = ([-100] * (seq_len - 1) + ids[(seq_len - 1):] + [-100] * (longest - ids_l))
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def run_chatglm():
    train_data = ("/data/zhangxin/code/LLM/glm/tmp_nlptask/LLM/chatglm/data/暖气不热_nohistory.json")

    max_seq_length = 512
    skip_overlength = False

    config = AutoConfig.from_pretrained(model_type, trust_remote_code=True)
    model = AutoModel.from_pretrained("/data/zhangxin/code/LLMModel/chatglm2-6b", trust_remote_code=True)


    dataset = read_jsonl(tokenizer, config, train_data, max_seq_length, skip_overlength)
    train_dataset = datasets.Dataset.from_dict(dataset)

    # 配置训练参数
    training_args = TrainingArguments(
        "/data/zhangxin/code/LLM/glm/tmp_nlptask/LLM/chatglm/output",
        fp16=True,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        max_steps=100,
        logging_steps=20,
        remove_unused_columns=False,
        seed=0,
        data_seed=0,
        group_by_length=False,
    )

    # 创建 ModifiedTrainer 对象并开始训练
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    run_chatglm()
