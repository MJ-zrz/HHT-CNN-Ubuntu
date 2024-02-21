import torch
import torch.nn as nn

from BasicBlock import *
from Bottleneck import *

# ResNet整个网络的框架部分
class ResNet(nn.Module):

    def __init__(self,
                 block,   # 残差结构，Basicblock or Bottleneck
                 blocks_num,   # 列表参数，所使用残差结构的数目，如对ResNet-34来说即是[3,4,6,3]
                 num_classes=1000,   # 训练集的分类个数
                 include_top=True):   # 为了能在ResNet网络基础上搭建更加复杂的网络，默认为True
        super(ResNet, self).__init__()
        self.include_top = include_top   # 传入类变量

        self.in_channel = 64   # 通过max pooling之后所得到的特征矩阵的深度

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)   # 输入特征矩阵的深度为3（RGB图像），高和宽缩减为原来的一半
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # 高和宽缩减为原来的一半

        self.layer1 = self._make_layer(block, 64, blocks_num[0])   # 对应conv2_x
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)   # 对应conv3_x
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)   # 对应conv4_x
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)   # 对应conv5_x

        if self.include_top:   # 默认为True
            # 无论输入特征矩阵的高和宽是多少，通过自适应平均池化下采样层，所得到的高和宽都是1
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)   # num_classes为分类类别数

        for m in self.modules():   # 卷积层的初始化操作
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):   # stride默认为1
        # block即BasicBlock/Bottleneck
        # channel即残差结构中第一层卷积层所使用的卷积核的个数
        # block_num即该层一共包含了多少层残差结构
        downsample = None

        # 左：输出的高和宽相较于输入会缩小；右：输入channel数与输出channel数不相等
        # 两者都会使x和identity无法相加
        if stride != 1 or self.in_channel != channel * block.expansion:  # ResNet-18/34会直接跳过该if语句（对于layer1来说）
            # 对于ResNet-50/101/152：
            # conv2_x第一层也是虚线残差结构，但只调整特征矩阵深度，高宽不需调整
            # conv3/4/5_x第一层需要调整特征矩阵深度，且把高和宽缩减为原来的一半
            downsample = nn.Sequential(       # 下采样
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))   # 将特征矩阵的深度翻4倍，高和宽不变（对于layer1来说）

        layers = []
        layers.append(block(self.in_channel,  # 输入特征矩阵深度，64
                            channel,  # 残差结构所对应主分支上的第一个卷积层的卷积核个数
                            downsample=downsample,
                            stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):   # 从第二层开始都是实线残差结构
            layers.append(block(self.in_channel,  # 对于浅层一直是64，对于深层已经是64*4=256了
                                channel))  # 残差结构主分支上的第一层卷积的卷积核个数
        
        # 通过非关键字参数的形式传入nn.Sequential
        return nn.Sequential(*layers)   # *加list或tuple，可以将其转换成非关键字参数，将刚刚所定义的一切层结构组合在一起并返回

# 正向传播过程
    def forward(self, x):
        x = self.conv1(x)   # 7×7卷积层
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # 3×3 max pool

        x = self.layer1(x)   # conv2_x所对应的一系列残差结构
        x = self.layer2(x)   # conv3_x所对应的一系列残差结构
        x = self.layer3(x)   # conv4_x所对应的一系列残差结构
        x = self.layer4(x)   # conv5_x所对应的一系列残差结构

        if self.include_top:
            x = self.avgpool(x)    # 平均池化下采样
            x = torch.flatten(x, 1)    
            x = self.fc(x)

        return x
    
    
def resnet18(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet18-5c106cde.pth    
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top) 
    
def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)    
    
    
