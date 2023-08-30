import threading


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def run_thread(thread_num, data_list):
    res_data = []
    threads = []
    tmp_num = len(data_list) // thread_num
    for i in range(thread_num):
        if i == thread_num - 1:
            data_tmp = data_list[i * tmp_num:]
        else:
            data_tmp = data_list[i * tmp_num:(i + 1) * tmp_num]
        t = MyThread(list_data_fun, args=(data_tmp,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
        res_data.extend(t.get_result())
    return res_data


def list_data_fun(num_list):
    res = []
    for num in num_list:
        res.append(num)
    return res


if __name__ == '__main__':
    thread_num = 5
    data_list = [i for i in range(100)]
    data = run_thread(thread_num, data_list)
    print(data)
