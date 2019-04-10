# -*- coding: utf-8 -*-
import torchvision as tv
import torchvision.transforms as transforms
import torch as t
from torchvision.transforms import ToPILImage
show=ToPILImage()       #把Tensor转成Image，方便可视化
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import os

GPUID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID
IS_DOWN_LOAD = True

###############数据加载与预处理
transform = transforms.Compose([transforms.ToTensor(),#转为tensor
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),#归一化
                                ])
#训练集
trainset=tv.datasets.CIFAR10(root='./cifar/train/',
                             train=True,
                             download=IS_DOWN_LOAD,
                             transform=transform)

trainloader=t.utils.data.DataLoader(trainset,
                                    batch_size=4,
                                    shuffle=True,
                                    num_workers=0)
#测试集
testset=tv.datasets.CIFAR10(root='./cifar/test/',
                             train=False,
                             download=IS_DOWN_LOAD,
                             transform=transform)

testloader=t.utils.data.DataLoader(testset,
                                   batch_size=4,
                                   shuffle=True,
                                   num_workers=0)


classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

(data,label)=trainset[100]
print(classes[label])

show((data+1)/2).resize((100,100))

# dataiter=iter(trainloader)
# images,labels=dataiter.next()
# print(''.join('11%s'%classes[labels[j]] for j in range(4)))
# show(tv.utils.make_grid(images+1)/2).resize((400,100))
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(trainloader)
images, labels = dataiter.next()
print("image size",images.size())
for l in labels:
    print(classes[l])
imshow(torchvision.utils.make_grid(images))
plt.show()#关掉图片才能往后继续算


#########################定义网络
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net=Net().cuda()
print(net)

#############定义损失函数和优化器
from torch import optim
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)

##############训练网络
from torch.autograd import Variable
import time


def run_test():
    right_num = 0;
    whole_num = 0
    for i,data in enumerate(testloader,0):
        inputs,labels=data
        inputs,labels=Variable(inputs.cuda()),Variable(labels.cuda())

        outputs=net(inputs)
        cpu_outputs=outputs.cpu()
        nd = cpu_outputs.detach().numpy()
        z = np.argmax(nd,axis=1)
        cpu_labels=labels.cpu()
        nl = cpu_labels.detach().numpy()
        for a in zip(z,nl):
            if a[0] == a[1]:
                right_num += 1
            whole_num += 1
    right_percent = float(right_num)/float(whole_num)
    print("whole_num(%d) right_num(%d) right_percent(%0.4f)"%(whole_num,right_num,right_percent))

# def find_max(d):
#     nd = d.detach().numpy()
#     z = np.argmax(nd,axis=1)
#     print(nd)

running_loss=0.0
start_time = time.time()
for epoch in range(2):

    for i,data in enumerate(trainloader,0):
        #输入数据
        inputs,labels=data
        inputs,labels=Variable(inputs.cuda()),Variable(labels.cuda())
        #梯度清零
        optimizer.zero_grad()

        outputs=net(inputs)

        loss=criterion(outputs,labels)
        loss.backward()
        #更新参数
        optimizer.step()

        # 打印log
        running_loss += loss.data[0]
        if i % 2000 == 1999:
            print('[%d,%5d] loss:%.3f' % (epoch + 1, i + 1, running_loss / 2000))
            run_test()
            running_loss = 0.0
print('finished training')
end_time = time.time()
print("Spend time:", end_time - start_time)
model_name = format("cifar_model_%0.4f.pkl"%(running_loss))
t.save(net,model_name)