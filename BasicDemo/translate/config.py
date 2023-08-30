import argparse

parser = argparse.ArgumentParser()

# 基本参数配置
parser.add_argument("-max_len", type=int, default=20)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-num_epochs", type=int, default=10)
parser.add_argument("-path_txt", type=str, default="/data/zhangxin/code/LLM/glm/tmp_nlptask/BasicDemo/translate/data/en_cn.txt")

args = parser.parse_args()
