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
    log_path = config["utils"]["log_save_path"]
    if args.delete_log:
        shutil.rmtree(log_path)
        os.mkdir(log_path)
    if args.log_to_file:
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

    return device


def save_model(Net, save_path, del_before):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model_name = time.strftime("AlexNet-%Y-%m-%d-%H-%M-%S.pth", time.localtime())
    save_file = os.path.join(save_path, model_name)

    if del_before:
        shutil.rmtree(save_path)
        os.mkdir(save_path)
        torch.save(Net.state_dict(), save_file)
    else:
        torch.save(Net.state_dict(), save_file)


class Tensorboard_settings():
    def __init__(self, work_path):
        self.workspace = work_path
        self.writer = SummaryWriter(self.workspace, flush_secs=1)
        self.workspace_create(self.workspace)

    def workspace_create(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
            os.mkdir(path)
        else:
            os.mkdir(path)

    def tensorboard_update(self, y, x):
        self.writer.add_scalar("Loss", y, x, display_name="Training")


    def tensorboard_save(self):
        self.writer.close()


def tensorboard_settings_val(args, val_path, net_results):
    config = load_config(args)
    tb_val_path = config["utils"]["tb_val_path"]
    if os.path.exists(tb_val_path):
        shutil.rmtree(tb_val_path)
        os.mkdir(tb_val_path)
    else:
        os.mkdir(tb_val_path)

    val_writer = SummaryWriter(tb_val_path)
    for index, path in enumerate(val_path):
        val_img = np.array(Image.open(path).convert("RGB").resize((448, 448))).astype(np.uint8)
        val_writer.add_image("Val_Images", val_img, global_step=index, dataformats="HWC")
        val_writer.add_scalar("0-Cats/1-Dogs", net_results[index], index, display_name="Val Results")
    val_writer.close()