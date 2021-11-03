__author__ = 'ck_ch'
# -*- coding: utf-8 -*-
# third-party library
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class cifa_data_set(data.Dataset):
    def __init__(self,data_path,is_train):
        self.data_path = data_path
        self.imgs = [os.path.join(data_path,img) for img in os.listdir(data_path)]
        self.imgs_num = len(self.imgs)
        np.random.seed(100)
        self.imgs = np.random.permutation(self.imgs)
        # 数据转换操作，测试验证和训练的数据转换有所区别
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        if is_train == True:
            self.transforms = T.Compose([
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        self.normalize
                    ])
        else:
            self.transforms = T.Compose([
                        T.ToTensor(),
                        self.normalize
                    ])

    def __getitem__(self, index):
        '''
        返回一张图片的数据
        '''
        img_path = self.imgs[index]
        label = img_path.split('\\')[-1].split('_')[1]
        label = np.int64(label)
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        '''
        返回数据集中所有图片的个数
        '''
        return len(self.imgs)