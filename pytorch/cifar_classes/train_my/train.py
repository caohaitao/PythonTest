__author__ = 'ck_ch'
# -*- coding: utf-8 -*-
from dataset import cifa_data_set
import os
import torch
import torch.nn as nn
import torch.onnx
from torchnet import meter
from torch.autograd import Variable
import time
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

pkl_name = "cifa_my.pkl"
EPOCH = 32
BATCH_SIZE = 100
train_data_path = r'E:\pythonCode\PythonTest\pytorch\cifar_classes\imgs_train'
test_data_path = r'E:\pythonCode\PythonTest\pytorch\cifar_classes\imgs_test'

def val(model,dataloader):
    model.eval()

    confusion_matrix = meter.ConfusionMeter(10)
    for ii,data in enumerate(dataloader):
        input,label = data
        val_input = Variable(input).cuda()
        # val_label = Variable(label.long()).cuda()

        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(),label.long())

    model.train()

    cm_value = confusion_matrix.value()
    accuracy = (cm_value[0][0] + cm_value[1][1]) / \
               (cm_value.sum())
    return confusion_matrix, accuracy

def save_lr(lr):
    f = open("lr.txt","w+")
    s = format("%0.7f"%lr)
    f.writelines(s)
    print("save lr(%0.7f) success"%lr)

def get_lr():
    if not os.path.exists("lr.txt"):
        print("not find lr.txt return 0.001")
        return 0.001
    else:
        f = open("lr.txt",'r')
        s = f.read()
        f.close()
        return float(s)

def train():
    print("train begin,cuda(%d)"%torch.cuda.is_available())
    if os.path.exists(pkl_name):
        cnn = torch.load(pkl_name).cuda()
        print("load model(%s) success"%pkl_name)
    else:
        cnn = Net().cuda()
        print("new model success")

    print(cnn)

    lr = get_lr()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    train_data = cifa_data_set(train_data_path,True)
    val_data = cifa_data_set(test_data_path,False)

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_data,
        BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(10)
    previous_loss = 1e100

    for epoch in range(EPOCH):
        loss_meter.reset()
        confusion_matrix.reset()

        for ii,(data,label) in enumerate(train_dataloader):
            input = Variable(data).cuda()
            target = Variable(label)

            optimizer.zero_grad()
            score = cnn(input)
            loss = loss_func(score.cpu(),target)
            loss.backward()
            optimizer.step()

            if ii % 100 == 0:
                print("epoch(%d) step(%d) loss(%0.6f)"%(epoch,ii,loss))
            # 更新统计指标以及可视化
            loss_meter.add(loss.detach().numpy())
            s = score.data
            t = target.data
            confusion_matrix.add(s, t)

        torch.save(cnn,pkl_name)
        print("save model(%s) success"%pkl_name)

        val_cm,val_accuracy = val(cnn,val_dataloader)
        print.info("epoch(%d) loss(%0.6f) val_accuracy(%0.4f)"
                    %(epoch,loss_meter.value()[0],val_accuracy))
        print.info("train_cm")
        print.info("\n%s"%confusion_matrix.value())
        print.info("val_cm")
        print.info("\n%s"%val_cm.value())

        if loss_meter.value()[0] > previous_loss:
            print.info("lr change from %0.4f -> %0.4f"%(lr,lr*0.95))
            lr = lr * 0.95
            save_lr(lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]

if __name__=='__main__':
    train()
