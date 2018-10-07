__author__ = 'ck_ch'
import torch
import matplotlib.pyplot as plt
from net import *

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)

net = Net(n_feature=1, n_hidden=10, n_output=1)
net.load_state_dict(torch.load('pwd_2_function_params.pkl'))
y = net(x)
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), y.data.numpy(), 'r-', lw=5)
plt.ioff()
plt.show()
