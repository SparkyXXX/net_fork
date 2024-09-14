import numpy as np
import os
import shutil
import yaml

from tensorboardX import SummaryWriter
from PIL import Image


class Tensorboard_settings():
    def __init__(self, use_flag, work_pth):
        self.use_flag = use_flag
        self.workspace = work_pth
        self.writer = SummaryWriter(self.workspace, flush_secs=1)

        # 初始化时，调用workspace创建空间
        self.workspace_create(self.workspace)

    # tensorboard路径创建和指定
    def workspace_create(self, path):
        # 判断该路径是否存在，如果存在，就清空，不存在就创建
        # 清空的目的是为了让tensorboard每次读入的文件都是最新的
        # 否则文件夹下太多events需要指定
        if os.path.exists(path):
            shutil.rmtree(path) 
            # 再重新创建一下
            os.mkdir(path)
        else:
            os.mkdir(path)    

    # tensorboard训练配置
    def tensorboard_update(self, loss, epoch):
        if self.use_flag:
            # 训练过程我们要体现目前迭代的数据集轮数和LOSS函数之间的关系
            # 首先获取LOSS
            self.writer.add_scalar('Loss', loss, epoch, display_name='Training')
        else:
            pass

    def tensorboard_save(self):
        self.writer.close()    


# tensorboard测试配置
def tensorboard_settings_test(args, test_img_path, net_results):
    # 读取config文件,以字典的形式存于config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
    # 获取test过程tensorboard数据的存储路径
    test_pth = config['tensorboard_params']['test_tpth']
    
    # 判断该路径是否存在，如果存在，就清空，不存在就创建
    # 清空的目的是为了让tensorboard每次读入的文件都是最新的
    # 否则文件夹下太多events需要指定
    if os.path.exists(test_pth):
        shutil.rmtree(test_pth) 
        # 再重新创建一下
        os.mkdir(test_pth)
    else:
        os.mkdir(test_pth)  
    
    # 定义test过程的写入过程
    test_writer = SummaryWriter(test_pth)

    # 测试过程我们需要验证网络计算结果和原图的关系
    # 首先先将测试的图片放到网页上
    for idx, pth in enumerate(test_img_path):
        #读取图片并转为RGB,注意需要转为numpy
        test_img = np.array(Image.open(pth).convert('RGB').resize((448,448))).astype(np.uint8)
        #在网页上添加图片
        test_writer.add_image('Test_Imgs', test_img, global_step=idx, dataformats='HWC')
        # 其次建立一个水平坐标为图像索引数，垂直坐标为识别结果的曲线表达
        test_writer.add_scalar('0-Cat/1-Dog', net_results[idx], idx, display_name='Test_Results')

    test_writer.close()