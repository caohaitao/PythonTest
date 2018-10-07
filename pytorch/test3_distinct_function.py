# -*- coding: utf-8 -*-
__author__ = 'ck_ch'
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from net import *

# torch.manual_seed(1)    # reproducibl
# make fake data
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

net = Net(n_feature=2, n_hidden=10, n_output=2)     # define the network
print(net)  # net architecture

# 传入 net 的所有参数, 学习率
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
# 算误差的时候, 注意真实值!不是! one-hot 形式的, 而是1D Tensor, (batch,)
# 但是预测值是2D tensor (batch, n_classes)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
plt.ion()   # something about plotting


plt.ion()   # something about plotting

for t in range(100):
    # 喂给 net 训练数据 x, 输出分析值
    out = net(x)                 # input x and predict based on x
    # 计算两者的误差
    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    # 清空上一步的残余更新参数值
    optimizer.zero_grad()   # clear gradients for next train
    # 误差反向传播, 计算参数更新值
    loss.backward()         # backpropagation, compute gradients
    # 将参数更新值施加到 net 的 parameters 上
    optimizer.step()        # apply gradients

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()