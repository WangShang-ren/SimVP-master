import os
import logging
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    # 1. 设置 Python 自带的随机模块种子
    # 作用：如果你用 random.randint() 或 random.shuffle()，结果会固定
    random.seed(seed)
    random.seed(seed)

    # 2. 设置 NumPy 的随机种子
    # 作用：如果你用 np.random.rand() 或数据处理时用了 numpy 的随机操作，结果会固定
    np.random.seed(seed)

    # 3. 设置 PyTorch CPU 的随机种子
    # 作用：控制 PyTorch 的权重初始化（如 torch.nn.Linear）、数据加载器的打乱等
    torch.manual_seed(seed)

    # 4. 设置 cuDNN 为确定性模式
    # 作用：这是针对 NVIDIA GPU 的加速库。
    # 默认情况下，cuDNN 为了追求极致速度，可能会使用一些“非确定性”的算法（结果会有微小差异）。
    # 设为 True 后，强制它使用结果完全一致的算法，但可能会稍微牺牲一点点速度。
    cudnn.deterministic = True


def print_log(message):
    print(message)
    logging.info(message)

def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)