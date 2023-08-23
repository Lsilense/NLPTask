import argparse

parser = argparse.ArgumentParser()

# 基本参数配置
parser.add_argument("-max_len", type=int, default=512)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-num_epochs", type=int, default=10)
parser.add_argument("-cuda", type=str, default="cuda:4")
parser.add_argument("-csv_path", type=str, default="/data/zhangxin/code/LLM/glm/tmp_nlptask/BasicDemo/classification/data/ChnSentiCorp_htl_all.csv")

# cnn参数配置
parser.add_argument("-cnn_embedding_dim", type=int, default=200)
parser.add_argument("-cnn_n_filters", type=int, default=100)
parser.add_argument("-cnn_filter_sizes", type=list, default=[3, 4, 5])
parser.add_argument("-cnn_output_dim", type=int, default=2)
parser.add_argument("-cnn_dropout", type=float, default=0.3)
parser.add_argument("-cnn_lr", type=float, default=0.00008)

# lstm参数配置
parser.add_argument("-lstm_embedding_dim", type=int, default=200)
parser.add_argument("-lstm_hidden_dim", type=int, default=200)
parser.add_argument("-lstm_num_layers", type=int, default=2)
parser.add_argument("-lstm_dropout", type=float, default=0.3)
parser.add_argument("-lstm_output_dim", type=int, default=2)
parser.add_argument("-lstm_fix_embedding", type=bool, default=True)
parser.add_argument("-lstm_lr", type=float, default=0.00008)


# bert_chinese参数配置
parser.add_argument(
    "-bert_roberta_name",
    type=str,
    default="hfl/chinese-roberta-wwm-ext",
    help=["bert-base-chinese", "hfl/chinese-roberta-wwm-ext"],
)
parser.add_argument("-bert_hidden_size", type=int, default=768)
parser.add_argument("-bert_output_dim", type=int, default=2)
parser.add_argument("-bert_dropout", type=float, default=0.5)
parser.add_argument("-bert_lr", type=float, default=0.00008)


args = parser.parse_args()
