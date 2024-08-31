"""
训练
"""

import argparse
import logging
import utils
import prepare
import models
from torch import nn
from torch import optim
from tqdm import tqdm


def train(args):
    # 读取参数
    config = utils.load_config(args)
    lr = config["run"]["lr"]
    train_epoches = config["run"]["train_epoches"]
    model_save_path = config["run"]["model_save_path"]
    tb_train_path = config["utils"]["tb_train_path"]
    TB = utils.TB_settings(work_path=tb_train_path)
    logging.info("Learning Rate: {}".format(lr))
    logging.info("Train Epoches: {}".format(train_epoches))
    logging.info("Model Save Path: {}".format(model_save_path))

    # 载入数据
    dataset = prepare.data_loader(config)
    train_loader = dataset["trainset"]

    # 实例化网络、损失函数、优化器
    MyNet = models.alexnet(num_classes=2)
    MyNet.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizor = optim.Adam(MyNet.parameters(), lr)

    # 开始训练
    count = 0
    for epoch in range(train_epoches):
        MyNet.train()
        running_loss = 0.0

        with tqdm(total=len(train_loader),
                  desc="epoch:{}".format(epoch),
                  bar_format="{desc} |{bar}| {elapsed}<{remaining}, {rate_fmt} {postfix}",
                  ncols=70) as t:
            for step, data in enumerate(train_loader, start=0):
                logging.info("-------------new batch---------------")
                # 读取数据，清零梯度，前向传播，计算损失，反向传播，更新梯度，损失累加
                imgs, labels = data
                # data为一个list，其中有两个tensor
                # 一个shape为torch.Size([100, 3, 224, 224])，表示一批100张图片
                # 另一个shape为torch.Size([100])，表示这100张图片对应的类别(0 or 1)
                # step 用于标记枚举的轮次
                
                optimizor.zero_grad()
                out = MyNet(imgs.to(device))
                loss = loss_func(out, labels.to(device))
                loss.backward()
                optimizor.step()
                running_loss += loss.item()

                # 更新显示
                t.set_postfix_str("Loss:{:^7.3f}".format(loss))
                t.update()
                TB.train_update(loss, count)
                TB.save()
                count += 1
            utils.save_model(MyNet, model_save_path, del_before=args.delete_model)
    tqdm.write("\nTrain Finish!\n")


if __name__ == "__main__":
    # 读取配置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="AlexNet script")
    parser.add_argument("--log_to_file", action="store_true", help="save log to txt file")
    parser.add_argument('--delete_model', action='store_true', help='delete old training results')
    parser.add_argument('--delete_log', action='store_true', help='delete old log files')
    args = parser.parse_args()
    device = utils.config(args)
    
    # 运行
    train(args)

