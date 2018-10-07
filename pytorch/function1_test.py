__author__ = 'ck_ch'
# -*- coding: utf-8 -*-
import os
# third-party library
import torch
import torch.nn as nn
import numpy as np
import os
from torch.autograd import Variable

# Hyper Parameters
 # 训练整批数据多少次, 为了节约时间, 我们只训练一次
EPOCH = 300              # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # 学习率

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.out = nn.Linear(1, 1)   # fully connected layer, output 10 classes

    def forward(self, x):
        s = x.size(0)
        #x= x.reshape(s,1)
        x = x.view(s,-1)
        output = self.out(x)
        return output    # return x for visualization

x = np.ndarray(shape=(3,1,1)).astype(np.float32)
x[0][0][0] = 1.0
x[1][0][0] = 2.0
x[2][0][0] = 3.0
y = 2*x
# tx = Variable(torch.from_numpy(x))
# ty = Variable(torch.from_numpy(y))
tx = torch.from_numpy(x)
ty = torch.from_numpy(y)
ty = ty.view(ty.size(0),-1)

cnn = CNN()
print(cnn)
optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()
for epoch in range(EPOCH):
    output = cnn(tx)
    loss = loss_func(output,ty)
    print('epoch=%d loss=%0.4f'%(epoch,loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# test_x = np.ndarray(shape=(9,1)).astype(np.float32)
# test_x[0] = 1.0
# test_x[1] = 2.0
# test_x[2] = 3.0
# test_x[3] = 4.0
# test_x[4] = 5.0
# test_x[5] = 6.0
# test_x[6] = 7.0
# test_x[7] = 8.0
# test_x[8] = 9.0
# torch_test_x = torch.from_numpy(test_x)
#
# prediction = cnn(torch_test_x)
# print(prediction)

torch.save(cnn,'function1.pkl')

dummy_init = torch.autograd.Variable(torch.randn(3,1,1))
torch.onnx.export(cnn,dummy_init,"function1.proto",verbose=True)