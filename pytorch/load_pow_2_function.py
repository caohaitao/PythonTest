__author__ = 'ck_ch'
import torch
import matplotlib.pyplot as plt
from ConvertModel import ConvertModel_ncnn

print(torch.__version__)

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)

net = torch.load('pow_2_function.pkl')
print(net)
y = net(x)
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), y.data.numpy(), 'r-', lw=5)
plt.ioff()
plt.show()

# inputShape = [1,1]
# text_net,binary_weights = ConvertModel_ncnn(
#     net,
#     y,
#     softmax = False
# )
