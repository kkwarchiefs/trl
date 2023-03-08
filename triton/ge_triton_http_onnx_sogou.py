# 需要安装以下环境，在线编码镜像 默认已经安装。
# pip3 install numpy
# pip3 install nvidia-pyindex
# pip3 install tritonclient[all]

import time

import numpy as np
import tritonclient.http as httpclient

model_name = "REL_model_onnx"  # 模型目录名/venus注册模型名称
address = "10.212.204.89:8000"  # 机器地址


def request(count):
    triton_client = httpclient.InferenceServerClient(url=address)

    # input = httpclient.InferInput('input', [1, 3, 224, 224], 'FP32')
    # data = np.random.rand(1, 3, 224, 224).astype(np.float32)  # 模型输入数据（array）

    # 模型输入数据（bytes）
    # 数据类型参考数据类型映射关系的API一列
    inputs = []
    inputs.append(httpclient.InferInput('input_ids', [1, 22], 'INT64'))
    inputs.append(httpclient.InferInput('position_ids', [1, 2, 22], 'INT64'))
    inputs.append(httpclient.InferInput('attention_mask', [1, 1, 22, 22], 'INT64'))
    data = np.array([[50002, 2705, 14364, 396, 16605, 44540, 43446, 1022, 4660, 50009,
                      50000, 50002, 2705, 14364, 396, 16605, 44540, 43446, 1022, 4660,
                      50009, 50000]], dtype=np.int64)
    data2 = np.array([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 9, 9, 9, 9, 9,
                        9, 9, 9, 9, 9],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6,
                        7, 8, 9, 10, 11]]], dtype=np.int64)
    data3 = np.array([[[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]], dtype=np.int64)
    inputs[0].set_data_from_numpy(data)
    inputs[1].set_data_from_numpy(data2)
    inputs[2].set_data_from_numpy(data3)
    output = httpclient.InferRequestedOutput('output')

    time_s = time.time()
    for _ in range(count):
        results = triton_client.infer(
            model_name,
            inputs,
            model_version='1',
            outputs=[output],
            request_id='1'
        )
        print(results.as_numpy('output'))
    time_used = time.time() - time_s
    return time_used


if __name__ == '__main__':
    request(1)
    # assert len(sys.argv) > 2
    # process_count = int(sys.argv[1])
    # request_count = int(sys.argv[2])
    #
    # start = time.time()
    # pool = mp.Pool(process_count)
    # ret_time = pool.map(
    #     request, [request_count for _ in range(process_count)], chunksize=1)
    # end = time.time()
    #
    # print("process cost:", ret_time)
    # print("throughput:",  request_count * process_count / (end-start), "qps")
    # print("latency:", np.sum(ret_time) /
    #       (process_count * request_count) * 1000, "ms")
