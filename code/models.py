"""
网络结构
"""

import torch.nn as nn

def alexnet(num_classes):
    return AlexNet(num_classes)

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        #######################第一层############################
        #输入图像尺寸为224*224，3通道，输出55*55，96个特征图
        self.Layer1_conv = nn.Conv2d(in_channels=3,
                                    out_channels=96,
                                    kernel_size=11,
                                    stride=4,
                                    padding=2,
                                    bias=False)                                   
        self.Layer1_bn = nn.BatchNorm2d(96)
        self.Layer1_relu = nn.ReLU()
        #池化层
        self.Layer1_mp = nn.MaxPool2d(kernel_size=3, stride=2)

        #######################第二层############################
        #输入图像尺寸27*27，96个特征图，输出13*13，256个特征图
        self.Layer2_conv = nn.Conv2d(in_channels=96,
                                    out_channels=256,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2,
                                    bias=False)
        self.Layer2_bn = nn.BatchNorm2d(256)
        self.Layer2_relu = nn.ReLU()
        #池化层
        self.Layer2_mp = nn.MaxPool2d(kernel_size=3, stride=2)

        #######################第三层############################
        #输入图像尺寸13*13, 256个特征图，输出13*13，384个特征图
        self.Layer3_conv = nn.Conv2d(in_channels=256,
                                    out_channels=384,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=False)
        self.Layer3_bn = nn.BatchNorm2d(384)
        self.Layer3_relu = nn.ReLU()
        #本层无池化

        #######################第四层############################
        #输入尺寸13*13，384个特征图，输出13*13，384个特征图
        self.Layer4_conv = nn.Conv2d(in_channels=384,
                                    out_channels=384,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=False)
        self.Layer4_bn = nn.BatchNorm2d(384)
        self.Layer4_relu = nn.ReLU()
        #本层无池化

        #######################第五层############################
        #输入尺寸13*13，384个特征图，输出6*6,256个特征图
        self.Layer5_conv = nn.Conv2d(in_channels=384,
                                    out_channels=256,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=False)
        self.Layer5_bn = nn.BatchNorm2d(256)
        self.Layer5_relu = nn.ReLU()
        #池化层
        self.Layer5_mp = nn.MaxPool2d(kernel_size=3, stride=2)
   
        #######################第六层############################
        #输入尺寸6*6,256个特征图（9216），输出尺寸4096
        self.Layer6_fc = nn.Linear(in_features=9216, 
                                out_features=4096,
                                bias=False)
        self.Layer6_bn = nn.BatchNorm1d(4096)
        self.Layer6_relu = nn.ReLU()
        #dropout
        self.Layer6_drop = nn.Dropout()

        #######################第七层############################
        #输入尺寸4096，输出尺寸4096
        self.Layer7_fc = nn.Linear(in_features=4096, 
                                out_features=4096,
                                bias=False)
        self.Layer7_bn = nn.BatchNorm1d(4096)
        self.Layer7_relu = nn.ReLU()
        #dropout
        self.Layer7_drop = nn.Dropout()

        #######################第八层############################
        #输入尺寸4096，输出尺寸100
        self.Layer8_fc = nn.Linear(in_features=4096, 
                                out_features=num_classes,
                                bias=True)

    #做一个特征图-全连接的Tensor尺度转换
    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features    

    #定义前向传播过程
    def forward(self, x):
        x = self.Layer1_conv(x)
        x = self.Layer1_bn(x)
        x = self.Layer1_relu(x)
        x = self.Layer1_mp(x)

        x = self.Layer2_conv(x)
        x = self.Layer2_bn(x)
        x = self.Layer2_relu(x)
        x = self.Layer2_mp(x)

        x = self.Layer3_conv(x)
        x = self.Layer3_bn(x)
        x = self.Layer3_relu(x)

        x = self.Layer4_conv(x)
        x = self.Layer4_bn(x)
        x = self.Layer4_relu(x)

        x = self.Layer5_conv(x)
        x = self.Layer5_bn(x)
        x = self.Layer5_relu(x)
        x = self.Layer5_mp(x)

        x = x.view(-1, self.num_flat_features(x))
        
        x = self.Layer6_fc(x)
        x = self.Layer6_bn(x)
        x = self.Layer6_relu(x)
        x = self.Layer6_drop(x)
        
        x = self.Layer7_fc(x)
        x = self.Layer7_bn(x)
        x = self.Layer7_relu(x)
        x = self.Layer7_drop(x)
        x = self.Layer8_fc(x)

        return x
