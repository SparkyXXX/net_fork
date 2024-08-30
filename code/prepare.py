"""
数据读取和预处理
"""

import random
import logging
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


class custom_val_dataset(Dataset):
    def __init__(self, root, transform):
        self.img_path = root
        self.transform = transform

    def __getitem__(self, index):
        path = self.img_path[index]
        img_origin = Image.open(path).convert("RGB")
        img = self.transform(img_origin)
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
