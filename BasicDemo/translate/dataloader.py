def read_txt(path):
    with open(path, "r", encoding="utf-8")as file:
        for line in file:
            line = line.strip().split("CC-BY")[0]
            line = line.split("  ")
            # line = [x if len(x)>1 ]
            print(line)
            

if __name__ == "__main__":
    path_txt = "/data/zhangxin/code/LLM/glm/tmp_nlptask/BasicDemo/translate/data/en_cn.txt"
    read_txt(path_txt)
