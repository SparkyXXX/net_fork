"""
Training script
This is the training script for AlexNet.

Author: Yu Gao
Date: 2021/01/08
"""
import argparse
import yaml
import os

import torch
import torch.nn as nn
import torch.optim as optim
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

def train(args):

    # 读取config文件,以字典的形式存于config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

    # 读取训练相关参数
    lr              = config['train_params']['learning_rate'] # 读取学习速率
    save_path       = config['train_params']['savepath'] #权重保存路径
    train_epoch     = config['train_params']['train_epoch']  #一共训练多少轮   
    tensorboard_pth = config['tensorboard_params']['train_tpth'] #获得训练过程的tensorboard路径
    use_tensorboard = args.tensorboard  #是否使用tensorboard

    # 保存/打印相关log信息
    logging.info('Learning Rate : {}'.format(lr))
    logging.info('Model Save Path : {}'.format(save_path))
    logging.info('Total Training Epoach : {}'.format(train_epoch))

    # 获取硬件设备信息
    device, device_id = device_detection(config)
    # 打印设备的硬件信息
    logging.info('Train device : {}-{}'.format('GPU' if str(device) == 'cuda' else 'CPU', device_id))

    # 获取数据集信息
    dataset = preparation.dataloader(config)
    train_loader = dataset['train']

    # 例化网络, 使用猫狗数据，num_classes = 2
    MyAlex = models.alexnet(num_classes=2)
    # 将模型放在GPU上
    MyAlex.to(device)

    # 优化器选择
    optimizer = optim.Adam(MyAlex.parameters(), lr=lr)
    best_acc = 0.0

    # 损失函数
    loss_func = losses.loss_function()
    # 保存/打印损失函数信息
    logging.info('Loss Function : {}'.format(loss_func))

    # 设置tensorboard工作路径
    TB = tensorboardsettings.Tensorboard_settings(use_tensorboard, tensorboard_pth)

    # 开始循环进行训练过程，共训练train_epoch轮,初始化一个计数器
    count = 0
    for epoch in range(train_epoch):
        # 告诉网络准备执行train操作，会有一些batch的优化操作
        MyAlex.train()
        running_loss = 0.0
        # 配置训练显示进度条   
        with tqdm(total = len(train_loader), 
                  desc='epoch:{}'.format(epoch), 
                  bar_format='{desc} |{bar}| {elapsed}<{remaining}, {rate_fmt} {postfix}', 
                  ncols=70) as t:

            # 开始每轮训练
            for step, data in enumerate(train_loader, start=0):
                # 注意此处，一定要深入了解train_loader中数据的具体存储内容
                # 取得图像和标签
                images, labels = data
                # 每次计算新的grad时，要把原来的grad清零
                optimizer.zero_grad()
                # 正向计算
                out = MyAlex(images.to(device))
                # 正向输出和标签进行误差计算
                loss = loss_func(out, labels.to(device))
                # 反向传播
                loss.backward()
                # 更新梯度信息
                optimizer.step()

                # 计算损失误差
                running_loss += loss.item()

                # 更新进度条
                t.set_postfix_str('Loss:{:^7.3f}'.format(loss))
                t.update()

                TB.tensorboard_update(loss, count)
                TB.tensorboard_save()
                # 计数器累加
                count += 1
            # 保存模型权重
            savemodel.save_model(MyAlex, save_path, args.delet)
            
    # 在tqdm下使用print会导致进度条重新打印，感兴趣的小伙伴可以试试       
    tqdm.write("\nTrain Finish!\n")

# 从此处开始阅读源码
if __name__ == '__main__':
    
    # 创建parser对象，为代码在Linux系统的运行提供指令输入接口
    parser = argparse.ArgumentParser(description = 'AlexNet train script')
    # 添加执行代码可调用的参数命令
    # 必加项：config文件
    # 可选项：log, tensorboard, del
    parser.add_argument('config', type=str,
                        help='config file of AlexNet')
    parser.add_argument('--log', action='store_true',
                        help='save log information to log.txt file')
    parser.add_argument('--tensorboard', action='store_true',
                        help='use tensorboard tools to show training results')
    parser.add_argument('--delet', action='store_true',
                        help='delete old training results')
    # 对添加的参数进行解析，可以理解为例化，写完这句上面添加的命令参数就能用了
    args = parser.parse_args()

    #hrx
    args.log = True

    # 更改显卡训练中你的进程名字，方便别人了解谁在使用GPU，要养成良好的习惯
    setproctitle.setproctitle("HuangRuixiang's Work! ^_^")

    #配置log的相关信息
    logconfig.log_config(args)

    # 开始训练过程
    train(args)
    