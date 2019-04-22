from model import resnet50
from dataset import DogCat
import os
import torch
import torch.nn as nn
import torch.onnx
from torchnet import meter
from torch.autograd import Variable
from log import *
import time

pkl_name = "cats_vs_dogs.pkl"
EPOCH = 32
BATCH_SIZE = 8
img_root = r'E:\tensorflow_datas\cats_vs_dogs\train'

logger = None

def val(model,dataloader):
    model.eval()

    confusion_matrix = meter.ConfusionMeter(2)
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
    global logger
    f = open("lr.txt","w+")
    s = format("%0.7f"%lr)
    f.writelines(s)
    logger.info("save lr(%0.7f) success"%lr)

def get_lr():
    global logger
    if not os.path.exists("lr.txt"):
        logger.info("not find lr.txt return 0.001")
        return 0.001
    else:
        f = open("lr.txt",'r')
        s = f.read()
        f.close()
        logger.info("get lr(%s) form lr.txt success"%s)
        return float(s)

def train():
    global logger
    logger.info("train begin,cuda(%d)"%torch.cuda.is_available())
    if os.path.exists(pkl_name):
        cnn = torch.load(pkl_name).cuda()
        logger.info("load model(%s) success"%pkl_name)
    else:
        cnn = resnet50().cuda()
        logger.info("new model success")

    print(cnn)

    lr = get_lr()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    train_data = DogCat(img_root,train=True)
    val_data = DogCat(img_root,train=False)

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
    confusion_matrix = meter.ConfusionMeter(2)
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
        logger.info("save model(%s) success"%pkl_name)

        val_cm,val_accuracy = val(cnn,val_dataloader)
        logger.info("epoch(%d) loss(%0.6f) val_accuracy(%0.4f)"
                    %(epoch,loss_meter.value()[0],val_accuracy))
        logger.info("train_cm")
        logger.info(val_cm.value())
        logger.info("val_cm")
        logger.info(confusion_matrix.value())

        if loss_meter.value()[0] > previous_loss:
            logger.info("lr change from %0.4f -> %0.4f"%(lr,lr*0.95))
            lr = lr * 0.95
            save_lr(lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]

if __name__=='__main__':
    logger = Logger(logname='cats_vs_dogs.log', loglevel=1, logger="fox").getlog()
    train()






