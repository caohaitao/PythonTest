__author__ = 'ck_ch'
# -*- coding: utf-8 -*-
import os
# third-party library
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from rect_pic_read import read_datas

EPOCH = 300              # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # 学习率

pkl_name = "rect_color.pkl"

class CNN(nn.Module):
    def __init__(self,width,height):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 160, 160)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 160, 160)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 80, 80)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 80, 80)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 80, 80)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 40, 40)
        )

        self.conv3 = nn.Sequential(         # input shape (32, 40, 40)
            nn.Conv2d(32, 64, 5, 1, 2),     # output shape (64, 40, 40)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (64, 20, 20)
        )

        out_one = int(int(width)/pow(2,3))
        self.out = nn.Linear(64 * out_one * out_one, 64)   # fully connected layer, output 10 classes
        self.out2 = nn.Linear(64 * out_one * out_one, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 128 * 10 * 10)
        output = self.out(x)
        output2 = self.out2(x)
        return output,output2    # return x for visualization

def train_model(width,height,torch_datas,torch_labels,torch_labels2):
    cnn = CNN(width,height)
    print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
    loss_func2 = torch.nn.MSELoss()
    for epoch in range(EPOCH):
        output,out_put2 = cnn(torch_datas)
        loss1 = loss_func(output,torch_labels)
        loss2 = loss_func2(out_put2,torch_labels2)
        loss = loss1+loss2
        if loss<0.001:
            break
        print('epoch=%d loss1=%0.4f,loss2=%0.4f,loss=%0.4f'%(epoch,loss1,loss2,loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(cnn,pkl_name)
    return cnn

def get_max_index(row):
    max_value = -999999999.0
    res = 0
    i = 0
    for a in row:
        if a>max_value:
            max_value = a
            res = i
        i = i+1
    return res

if __name__ == "__main__":
    datas,labels,labels2,w,h=read_datas()
    torch_datas = torch.from_numpy(datas)
    torch_labels = torch.from_numpy(labels)
    torch_labels2 = torch.from_numpy(labels2)

    if not os.path.exists(pkl_name):
        cnn = train_model(w,h,torch_datas,torch_labels,torch_labels2)
    else:
        cnn = torch.load(pkl_name)

    out_put,out_put2 = cnn(torch_datas)

    count = len(out_put)
    for i in range(count):
        max_index = get_max_index(out_put[i])
        s = format("%d-%d,%0.2f-%0.2f"%(torch_labels[i],max_index,torch_labels2[i],out_put2[i]))
        print(s)


