"""
训练、测试
"""

import argparse
import logging
import utils
import prepare
import models
import torch
from torch import nn
from torch import optim
from tqdm import tqdm


def train(args):
    # 读取参数
    config = utils.load_config(args)
    lr = config["run"]["lr"]
    train_epoches = config["run"]["train_epoches"]
    model_save_path = config["run"]["model_save_path"]
    tb_workspace = config["utils"]["tb_train_path"]
    use_tb = args.tensorboard
    TB = utils.Tensorboard_settings(use_flag=use_tb, work_path=tb_workspace)
    logging.info("Learning Rate: {}".format(lr))
    logging.info("Train Epoches: {}".format(train_epoches))
    logging.info("Model Save Path: {}".format(model_save_path))

    # 载入数据
    dataset = prepare.data_loader(config)
    train_loader = dataset["trainset"]

    # 实例化网络、损失函数、优化器
    MyNet = models.alexnet(num_class=2)
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
                # 读取数据，清零梯度，前向传播，计算损失，反向传播，更新梯度，损失累加
                imgs, labels = data
                optimizor.zero_grad()
                out = MyNet(imgs.to(device))
                loss = loss_func(out, labels.to(device))
                loss.backward()
                optimizor.step()
                running_loss += loss.item()

                # 更新显示
                t.set_postfix_str("Loss:{:^7.3f}".format(loss))
                t.update()
                TB.tensorboard_update(loss, epoch)
                TB.tensorboard_save()
                count += 1
            utils.save_model(MyNet, model_save_path, del_before=args.delete)
    tqdm.write("\nTrain Finish!\n")


def val(args):
    # 读取参数
    config = utils.load_config(args)
    model_weight_path = config["run"]["model_weight_path"]
    logging.info("Weight Path: {}".format(model_weight_path))

    # 载入数据
    dataset = prepare.data_loader(config)
    val_loader = dataset["valset"]

    # 实例化网络并载入
    MyNet = models.alexnet(num_class=2)
    MyNet.load_state_dict(torch.load(model_weight_path, map_location=device))
    MyNet.eval()

    # 开始前向传播训练
    with torch.no_grad():
        predict_results = []
        predict_class = {}

        with tqdm(total=len(val_loader),
                  desc="Testing",
                  bar_format="{desc} |{bar}| {elapsed}<{remaining}, {rate_fmt} {postfix}",
                  ncols=70) as t:
            for index, data in enumerate(val_loader):
                # 读取数据，预测
                img, label = data
                output = torch.softmax(torch.squeeze(MyNet(img)), dim=0)
                predict = torch.argmax(output).numpy()
                predict_results.append(predict)
                predict_class["Img" + str(index)] = "Cat" if predict == 0 else "Dog"

                # 更新显示
                t.update()
            utils.tensorboard_settings_val(args, dataset["val_img"], predict_results)
            logging.info("Predict Results: {}".format(predict_class))



if __name__ == "__main__":
    # 读取配置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="AlexNet script")
    parser.add_argument("--log", action="store_true", help="save log to txt file")
    parser.add_argument('--tensorboard', action='store_true', help='tensorboard to show training results')
    parser.add_argument('--delete', action='store_true', help='delete old training results')
    args = parser.parse_args()
    process, device = utils.config(args)
    
    # 运行
    if process == "train":
        train(args)
    elif process == "val":
        val(args)
