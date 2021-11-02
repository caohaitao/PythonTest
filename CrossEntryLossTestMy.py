# -*- coding: utf-8 -*-
#这个测试为了测试一次交叉熵优化概率的变化，一次交叉熵之后，训练数据的概率发生
#变化，变化的大小跟lr有关，也可以调整log前面的系数，例如概率增加，损失的概率
#会平均分配到其他的项上面，概率减少也是如此，不过一个数据的优化会对其他数据
#产生波及，其他没有训练的数据概率也会发生相应的变化
__author__ = 'ck_ch'
import torch
import torch.nn as nn
import torch.nn.functional as F

#随便定义一个网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(5,200),
            nn.ReLU(),
            nn.Linear(200,4)
        )

    def forward(self, x):
        out = self.fc(x)
        return out

#测试数据，为了测试一次交叉熵优化后概率的变化
di = torch.tensor([
    [0.1,0.2,0.3,0.4,0.5],
    [0.1,0.2,0.5,0.4,0.5],
    [0.433,0.298,0.378,0.192,0.256],
    [0.4,0.2,0.7,0.9,0.8],
    [0.3,0.7,0.9,0.1,0.44],
    [0,0,0,0,1]
]
)

net = Net()
optimizer = torch.optim.Adam(net.parameters(),lr=0.0001)

a = net(di)
a = F.softmax(a,dim=1)
print(a)

train_i = torch.tensor([
    [0.1,0.2,0.3,0.4,0.5]
]
)

label_i = torch.tensor([2],dtype=torch.int64)
for i in range(5):
    a2 = net(train_i)
    soft_a2 = F.softmax(a2,dim=1)
    sel = torch.index_select(soft_a2,1,label_i)
    loss = -1*torch.log(sel)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    a = net(di)
    a = F.softmax(a,dim=1)
    print(a)