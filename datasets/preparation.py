"""
Dataloader script
This is the dataset script for AlexNet.

Author: Yu Gao
Date: 2021/01/08
"""
import torch
import logging
import random
import numpy as np

from torchvision import transforms, datasets
from pathlib import Path
from PIL import Image

# 因为验证集需要自己配置，所以需要自己构造一个dataset类的生成函数，继承Dataset
class customdatasets(torch.utils.data.Dataset):
    def __init__(self, img_pth, transform=None):
        self.img_path = img_pth
        self.transform = transform

    # 重点编写__getitem__()
    def __getitem__(self, index):
        # 路径好理解
        path = self.img_path[index]
        # 因为测试集没有label，所以我们将索引号作为label
        # 最后验证结果是交给人来识别的，这个label就是一个序号的作用
        label = int(index)
        # 打开图片进行处理,转为RGB
        img_ori = Image.open(path).convert('RGB')
        # 如果格式变换函数不是None，就进行相应的格式变换
        if self.transform is not None:
            img = self.transform(img_ori)
        return img, label
    
    # 数据集长度
    def __len__(self):
        return len(self.img_path)

# 因为猫狗数据集的验证集没有标签（除非提前人工分好，或者从训练集里抠图）
# 所以只能随机选出一些图，然后人为验证，该函数的功能就是随机挑一部分验证图
def val_choice(val_path, val_numb):
    # 获取测试文件夹下的所有图片，并且打乱顺序,并且截取前val_numb个数据
    image_paths = list(Path(val_path).iterdir())    
    random.shuffle(image_paths)
    image_paths  = image_paths[:val_numb]
    return image_paths
    
# 数据集载入函数，包括图片从读取到数据集的构成
def dataloader(config):
    # 读取配置参数
    train_path = config['data']['root_train']
    val_path   = config['data']['root_val']
    batch_size = config['data']['batch_size']
    test_batch = config['data']['test_batch']
    val_numb   = config['test_params']['test_numb']

    # 数据格式转换功能
    data_transform = {
    # 训练集操作：
    # 1. 图片大小变为(224,224)
    # 2. 随机水平翻转，默认概率0.5
    # 3. 转换为tensor
    # 4. 标准化
    "train": transforms.Compose([transforms.Resize((224,224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    # 训练集操作： 
    # hrx：验证集操作？
    # 1. 图片大小变为(224,224)
    # 2. 转换为tensor
    # 3. 标准化
    "val": transforms.Compose([transforms.Resize((224,224)),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 读取指定路径下的所有图片，并形成数据集格式，每个文件夹会被标记成一个label
    # 每个文件下下的图片具有其文件夹的label
    train_dataset = datasets.ImageFolder(train_path, transform=data_transform["train"])
    # 存入验证数据,首先先选择val_numb个图片作为验证
    choice_path = val_choice(val_path, val_numb)
    val_dataset = customdatasets(choice_path, transform=data_transform["val"])
    # 打印训练集和测试集的长度，确保没有问题
    logging.info('Trainset Numb : {}, Valset Numb : {}'.format(len(train_dataset), len(val_dataset)))

    #将数据集转化为可以迭代索引的格式
    train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0
                        )

    #将数据集转化为可以迭代索引的格式
    val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=test_batch,
                        shuffle=False,
                        num_workers=0
                        )

    dataset = {'train':train_loader,'val':val_loader, 'img_pth': choice_path}

    return dataset

