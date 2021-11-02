# -*- coding: utf-8 -*-
import torchvision as tv
import torchvision.transforms as transforms
import torch as t
from torchvision.transforms import ToPILImage
show=ToPILImage()       #把Tensor转成Image，方便可视化
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import os
import cv2
from PIL import Image

###############数据加载与预处理
transform = transforms.Compose([transforms.ToTensor(),#转为tensor
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),#归一化
                                ])
#训练集
trainset=tv.datasets.CIFAR10(root='./cifar/train/',
                             train=True,
                             download=False,
                             transform=None)


classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

s = trainset.__len__()
for i in range(s):
    (d,l) = trainset[i]
    p = format("image\\%d.jpg"%(i))
    d.save(p)
