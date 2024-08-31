"""
测试
"""

import argparse
import logging
import utils
import prepare
import models
import torch
from tqdm import tqdm


def val(args):
    # 读取参数
    config, device = utils.config(args)
    model_weight_path = config["run"]["model_weight_path"]
    tb_val_path = config["utils"]["tb_val_path"]
    TB = utils.TB_settings(tb_val_path)
    logging.info("Weight Path: {}".format(model_weight_path))

    # 载入数据
    dataset = prepare.data_loader(config)
    val_loader = dataset["valset"]

    # 实例化网络并载入
    MyNet = models.alexnet(num_classes=2)
    MyNet.load_state_dict(torch.load(model_weight_path, map_location=device))
    MyNet.eval()

    # 开始前向传播测试
    with torch.no_grad():
        predict_results = []
        predict_class = {}

        with tqdm(total=len(val_loader),
                  desc="Testing",
                  bar_format="{desc} |{bar}| {elapsed}<{remaining}, {rate_fmt} {postfix}\n",
                  ncols=70) as t:
            for index, data in enumerate(val_loader):
                # 读取数据，预测
                img, label = data
                output = torch.softmax(torch.squeeze(MyNet(img)), dim=0)
                predict = torch.argmax(output).numpy() # argmax返回最大值的索引
                predict_results.append(predict)
                predict_class["Img" + str(index)] = "Cat" if predict == 0 else "Dog"
                t.update()
        TB.val_update(dataset["val_img"], predict_results)
        TB.save()
        logging.info("Predict Results: {}".format(predict_class))


if __name__ == "__main__":
    # 读取配置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="AlexNet script")
    parser.add_argument("--log_to_file", action="store_true", help="save log to txt file")
    parser.add_argument('--delete_past_log', action='store_true', help='delete old log files')
    args = parser.parse_args()
    
    # 运行
    val(args)
