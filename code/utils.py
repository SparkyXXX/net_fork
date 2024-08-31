"""
日志输出、保存模型、tensorboard配置
"""
import yaml
import os
import time
import shutil
import logging
import setproctitle
import torch
import numpy as np
from tensorboardX import SummaryWriter
from PIL import Image


def load_config(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def config(args):
    setproctitle.setproctitle("HuangRuixiang is working ^_^")
    config = load_config(args)

    # 日志配置
    # logging.getLogger("PIL").setLevel(logging.ERROR)
    log_path = config["utils"]["log_save_path"]
    if args.delete_log:
        shutil.rmtree(log_path)
        os.mkdir(log_path)
    if args.log_to_file:
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        ticks = time.asctime(time.localtime(time.time()))
        ticks = str(ticks).replace(' ', '-').replace(':', '-')
        log_name = "{}.log".format(os.path.join(log_path, ticks))
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

    return device


def save_model(Net, save_path, del_before):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    ticks = time.strftime("AlexNet-%Y-%m-%d-%H-%M-%S.pth", time.localtime())
    model_name = os.path.join(save_path, ticks)

    if del_before:
        shutil.rmtree(save_path)
        os.mkdir(save_path)
    torch.save(Net.state_dict(), model_name)


class TB_settings():
    def __init__(self, work_path):
        self.workpath = work_path
        if os.path.exists(work_path):
            shutil.rmtree(work_path)
        os.mkdir(work_path)
        self.writer = SummaryWriter(self.workpath, flush_secs=1)

    def train_update(self, y, x):
        self.writer.add_scalar("Loss", y, x, display_name="Training")

    def val_update(self, val_img_path, net_results):
        for index, path in enumerate(val_img_path):
            val_img = np.array(Image.open(path).convert("RGB").resize((448, 448))).astype(np.uint8)
            self.writer.add_image("Val_Images", val_img, global_step=index, dataformats="HWC")
            self.writer.add_scalar("0-Cats/1-Dogs", net_results[index], index, display_name="Val Results")

    def save(self):
        self.writer.close()
