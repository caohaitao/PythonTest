__author__ = 'ck_ch'
# -*- coding: utf-8 -*-
# third-party library
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T


pkl_name = "cats_vs_dogs.pkl"

gmean = [0.40041953, 0.35345662, 0.32371968]
gstd = [0.23473245, 0.24891363, 0.24843082]

class DogCat(data.Dataset):
    def __init__(self,root,transforms=None,train=True,test=False):
        self.test = test
        imgs = [os.path.join(root,img) for img in os.listdir(root)]

        imgs_num = len(imgs)

        np.random.seed(100)
        imgs = np.random.permutation(imgs)

        # 划分训练、验证集，验证:训练 = 3:7
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        if transforms is None:

            # 数据转换操作，测试验证和训练的数据转换有所区别
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            # 测试集和验证集不用数据增强
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
                # 训练集需要数据增强
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        '''
        返回一张图片的数据
        对于测试集，没有label，返回图片id，如1000.jpg返回1000
        '''
        img_path = self.imgs[index]
        # if self.test:
        #     label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        # else:
        #     label = 1 if 'dog' in img_path.split('/')[-1] else 0

        label = img_path.split('\\')[-1].split('.')[0].split('_')[1]
        label = np.int64(label)

        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        '''
        返回数据集中所有图片的个数
        '''
        return len(self.imgs)
