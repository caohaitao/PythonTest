__author__ = 'ck_ch'
import torch
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)

net = torch.load('pow_2_function.pkl')
y = net(x)
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), y.data.numpy(), 'r-', lw=5)
plt.ioff()
plt.show()