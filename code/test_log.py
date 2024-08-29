"""
数据读取和预处理
"""

import argparse
import random
import logging
from PIL import Image
from pathlib import Path
import torch
import yaml
import os
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets



def load_config(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def config(args):
    config = load_config(args)

    # 日志配置
    save = args.log
    if save:
        save_path = config["utils"]["log_save_path"]
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        ticks = time.asctime(time.localtime(time.time()))
        ticks = str(ticks).replace(' ', '-').replace(':', '-')
        log_name = "{}.log".format(os.path.join(save_path, ticks))
        logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', 
                            datefmt='%Y/%m/%d %H:%M:%S', 
                            level=logging.DEBUG,
                            filemode='a',
                            filename=log_name)
    else:
        logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', 
                            datefmt='%Y/%m/%d %H:%M:%S', 
                            level=logging.DEBUG)
        
    # 设备配置
    dev    = config["utils"]["dev"]
    dev_id = config["utils"]["dev_id"]
    device = torch.device(dev)
    if dev == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
    logging.info("Device: {}-{}".format("GPU" if str(device) == "cuda" else "CPU", dev_id))
    
    # 任务选择
    process = config["utils"]["process"]
    logging.info("Task: {}".format("Training" if str(process) == "train" else "Validating"))

    return process, device



class custom_val_dataset(Dataset):
    def __init__(self, root, transform):
        self.img_path = root
        self.transform = transform

    def __getitem__(self, index):
        path = self.img_path[index]
        img_origin = Image.open(path).convert("RGB")
        img = img_origin.transform(img_origin)
        label = int(index)
        return img, label
    
    def __len__(self):
        return len(self.img_path)


def data_loader(config):
    # 读取参数
    train_path       = config["data"]["train_path"]
    val_img_path1    = config["data"]["val_path1"]
    val_img_path2    = config["data"]["val_path2"]
    train_batch_size = config["data"]["train_batch_size"]
    val_batch_size   = config["data"]["val_batch_size"]
    val_num          = config["data"]["val_num"]

    # 格式转换
    data_transform = {
        "train": transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 生成验证集
    val_path = list(Path(val_img_path1).iterdir())
    temp_path = list(Path(val_img_path2).iterdir())
    val_path.extend(temp_path)
    random.shuffle(val_path)
    val_path = val_path[:val_num]

    # 生成数据集，完成标注
    train_dataset = datasets.ImageFolder(root=train_path, transform=data_transform["train"])
    val_dataset   = custom_val_dataset(root=val_path, transform=data_transform["val"])
    logging.info("Trainset Len: {}, Valset Len: {}".format(len(train_dataset), len(val_dataset)))

    # 产生迭代器
    train_loader = DataLoader(train_dataset, train_batch_size, shuffle=True,  num_workers=16)
    val_loader   = DataLoader(val_dataset,   val_batch_size,   shuffle=False, num_workers=16)

    dataset = {'trainset':train_loader,'valset':val_loader, 'val_img': val_path}
    return dataset
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="AlexNet script")
    parser.add_argument("--log", action="store_true", help="save log to txt file")
    args = parser.parse_args()
    process, device = config(args)
    my_config = load_config(args)
    
    dataset = data_loader(my_config)
    train_loader = dataset["trainset"]
    for step, data in enumerate(train_loader, start=0):
        logging.info("_____________{}________________".format(step))
        logging.info("_____________{}________________".format(data[0].shape))
        logging.info("_____________{}________________".format(data[1].shape))