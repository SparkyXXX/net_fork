"""
Test script
This is the test script for AlexNet.

Author: Yu Gao
Date: 2021/01/08
"""
import argparse
import yaml
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import setproctitle
import logging

from models import models
from datasets import preparation
from loss import losses
from utils import logconfig
from utils import savemodel
from utils import tensorboardsettings
from tqdm import tqdm

# 指训练过程使用的硬件设备
def device_detection(config):
    dev    = config['device']['dev']   # 读取device设备信息
    dev_id = config['device']['id']    # 读取device设备信息

    if(dev != 'cuda'):
        device = torch.device(dev)
    else:
        #指定使用的GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
        device = torch.device(dev)

    return device, dev_id

# 测试过程代码
def test(args):
    # 读取config文件,以字典的形式存于config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

    # 读取训练相关参数
    model_weight_path = config['test_params']['model_weight_path'] # 读取权重保存路径

    # 保存/打印相关log信息
    logging.info('Weight Path : {}'.format(model_weight_path))
    
    # 获取硬件设备信息
    device, device_id = device_detection(config)
    # 打印设备的硬件信息
    logging.info('Test device : {}-{}'.format('GPU' if str(device) == 'cuda' else 'CPU', device_id))

    # 获取数据集信息
    dataset = preparation.dataloader(config)
    test_loader = dataset['val']

    # 例化网络, 使用猫狗数据，num_classes = 2
    MyAlex = models.alexnet(num_classes=2)
    # 载入训练的网络模型
    MyAlex.load_state_dict(torch.load(model_weight_path, map_location=device))
    # 告诉网络准备执行val操作
    MyAlex.eval()

    # 前向传播训练
    with torch.no_grad():
        predict_results = []
        predict_class = {}
        # 配置训练显示进度条 
        with tqdm(total = len(test_loader), 
        desc='Testing:', 
        bar_format='{desc} |{bar}| {elapsed}<{remaining}, {rate_fmt} {postfix}', 
        ncols=70) as t:

            # 开始循环进行图片测试，有多少张图片，训练多少轮
            for index, test_img in enumerate(test_loader):
                # 注意此处，一定要深入了解test_loader中数据的具体存储内容
                T_img, T_label = test_img
                # 注意理解这里为什么要加入torch.squeeze，需要降低维度
                output = torch.squeeze(MyAlex(T_img))
                predict = torch.softmax(output,dim = 0)
                predict_cla = torch.argmax(predict).numpy()
                predict_results.append(predict_cla)
                predict_class['Img' + str(index)] = 'Cat' if predict_cla == 0 else 'Dog'
                t.update()

        tensorboardsettings.tensorboard_settings_test(args, dataset['img_pth'], predict_results)

        # 保存/打印相关log信息
        logging.info('Dog or Car : {}'.format(predict_class))
        

# 从此处开始阅读源码
if __name__ == '__main__':
    
    # 创建parser对象，为代码在Linux系统的运行提供指令输入接口
    parser = argparse.ArgumentParser(description = 'AlexNet test script')
    # 添加执行代码可调用的参数命令
    # 必加项：config文件
    # 可选项：log
    parser.add_argument('config', type=str,
                        help='config file of AlexNet')
    parser.add_argument('--log', action='store_true',
                        help='save log information to log.txt file')
    # 对添加的参数进行解析，可以理解为例化，写完这句上面添加的命令参数就能用了
    args = parser.parse_args()

    # hrx
    args.log = True

    # 更改显卡训练中你的进程名字，方便别人了解谁在使用GPU，要养成良好的习惯
    setproctitle.setproctitle("HuangRuixiang's Work! ^_^")

    # 配置log的相关信息
    logconfig.log_config(args)

    # 开始训练过程
    test(args)


    
