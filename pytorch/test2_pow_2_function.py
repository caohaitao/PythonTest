__author__ = 'ck_ch'
# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt
from net import *
import numpy as np

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
#y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
y = 2*x

net = Net(n_feature=1, n_hidden=1, n_output=1)

print(net)

# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)

plt.ion()   # something about plotting

for t in range(500):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    print("loss=%0.6f"%loss)

    optimizer.zero_grad()   # clear gradients for next train

    loss.backward()         # backpropagation, compute gradients

    optimizer.step()        # apply gradients

    # if t % 5 == 0:
    #
    #     # plot and show learning process
    #
    #     plt.cla()
    #
    #     plt.scatter(x.data.numpy(), y.data.numpy())
    #
    #     plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    #
    #     plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
    #
    #     plt.pause(0.1)

# plt.ioff()
#
# plt.show()

test_x = np.ndarray(shape=(3,1)).astype(np.float32)
test_x[0] = 2.0
test_x[1] = 3.0
test_x[2] = 4.0
torch_test_x = torch.from_numpy(test_x)

prediction = net(torch_test_x)

torch.save(net,'pow_2_function.pkl')
torch.save(net.state_dict(),'pwd_2_function_params.pkl')

dummy_init = torch.autograd.Variable(torch.randn(1,1))
torch.onnx.export(net,dummy_init,"pow_2_function.proto",verbose=True)