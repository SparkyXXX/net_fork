import time
import os
import torch
import shutil

# 根据时间，对训练模型的权重文件进行命名, 每次保存最新的权重文件，删除之前保存的权重文件(默认为True)
def save_model(Net, save_path, del_before = True):
    # 获取当时的时间，作为模型权重的名字，名字格式：模型-年-月-日-时-分-秒.pth
    # strftime为格式化日期函数
    model_name = time.strftime("AlexNet-%Y-%m-%d-%H-%M-%S.pth", time.localtime())
    savefile = os.path.join(save_path, model_name)
    
    # 判断一下路径是否存在
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 判断之前是否有存储的非最新结果
    if del_before:
        # 能删除该文件夹和文件夹下所有文件
        shutil.rmtree(save_path) 
        # 再重新创建一下
        os.mkdir(save_path)
        # 保存当前的模型
        torch.save(Net.state_dict(), savefile)
    # 如果不删除文件，就一直保存下去
    else:
        torch.save(Net.state_dict(), savefile)

