import re
import json
import threading
import pandas as pd
import requests


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result


def run_thread(thread_num, data_list):
    res_data = []
    threads = []
    tmp_num = len(data_list) // thread_num
    for i in range(thread_num):
        if i == thread_num - 1:
            data_tmp = data_list[i * tmp_num:]
        else:
            data_tmp = data_list[i * tmp_num:(i + 1) * tmp_num]
        print("第{}个进程数据长度：{}".format(i, len(data_tmp)))
        t = MyThread(get_test_res, args=(data_tmp, ))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
        res_data.extend(t.get_result())
    return res_data


def get_res_third(path):
    res_list_third = []
    for sheetName in pd.read_excel(path, sheet_name=None).keys():
        print(sheetName)
        datas = pd.read_excel(path, sheet_name=sheetName)
        for i in range(0, len(datas["场景、问题"]), 2):
            res_dic = {
                "group": datas["场景、问题"][i],
                "问题": datas["场景、问题"][i + 1],
                "标签": datas["场景、问题"][i],
            }
            res_list_third.append(res_dic)
    return res_list_third


def get_test_res(res_list):

    url = 'http://127.0.0.1:8000/chat/'

    res_test_list = []
    question = set()
    for data in res_list:
        if data["问题"] in question:
            continue
        question.add(data["问题"])
        if not isinstance(data["问题"], str):
            continue

        params = {"query": data["问题"], "history": [], }
        html = requests.post(url, json.dumps(params))
        responses = json.loads(html.text)
        response = responses["response"]

        response_all = response.split("\n")[0]
        extract_response = re.findall(r"#<(.+?)>", response)

        if len(extract_response) == 0:
            extract_response = response_all
        else:
            extract_response = "#".join(extract_response)

        extract_response = extract_response.replace(" ", "")
        response_all = response_all.replace(" ", "")

        sample_dic = {
            "group": data["group"],
            "问题": data["问题"],
            "提取结果": extract_response,
            "完整结果": response_all,
        }
        print("问题长度{}。".format(len(question)), sample_dic)
        res_test_list.append(sample_dic)
    return res_test_list


if __name__ == "__main__":

    path_third = "/data/zhangxin/code/LLM/glm/test/data/excel/测试数据第三批.xlsx"

    res_list_third = get_res_third(path_third)

    print("第三批测试")
    thread_num = 8
    res_test_list_third = run_thread(thread_num, res_list_third)
    js_third = pd.DataFrame(res_test_list_third)
    js_third.to_excel("/data/zhangxin/code/LLM/glm/test/data/excel/test_第三批_newmodel.xlsx", index=False)
