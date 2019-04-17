import torch.nn as nn
import math
import os
from PIL import Image
import torchvision
import numpy as np
import random
import torch
import torch.utils.data as tdata

pkl_name = "class_renet18.pkl"
#32x32的图片太小，有bug，需要修改forward里面的参数，已经知道怎么用了，后面要用的话再说

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18():
    model = ResNet(BasicBlock,[2,2,2,2],10)
    return model

def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3], 10)
    return model


EPOCH = 1000              # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 10
LR = 0.01              # 学习率


gmean = [0.40041953, 0.35345662, 0.32371968]
gstd = [0.23473245, 0.24891363, 0.24843082]

def read_one_pic(pic_path):
    image = Image.open(pic_path)
    img = torchvision.transforms.ToTensor()(image)
    return img,img.size()[1],img.size()[2],img.size()[0]

def get_label_from_name(file_name):
    fname = file_name[0:-4]
    fs = fname.split('_')
    if len(fs) != 3:
        return None
    return fs[1],fs[2]


def read_one_dir_datas(dir,datas_out,nums):
    file_lists = []
    for (root,dirs,files) in os.walk(dir):
        for item in files:
            file_path = os.path.join(dir,item)
            file_lists.append([file_path,item])

    if nums is not None:
        file_lists = random.sample(file_lists,nums)

    for f in file_lists:
        file_path = f[0]
        item = f[1]
        l1, l2 = get_label_from_name(item)
        img, w, h, c = read_one_pic(file_path)
        for t, m, s in zip(img, gmean, gstd):
            t.sub_(m).div_(s)
        datas_out.append([img, w, h, c, l1, l2])


def read_whole_datas(root_dir,nums):
    pdatas = []
    read_one_dir_datas(root_dir,pdatas,nums)

    res = np.ndarray(shape=(len(pdatas),pdatas[0][3],pdatas[0][1],pdatas[0][2]),dtype=np.float32)
    labels = np.ndarray(shape=(len(pdatas)),dtype=np.int64)
    for i,d in enumerate(pdatas):
        res[i] = d[0]
        labels[i] = d[4]
    return res,labels

def train_model():
    print("cuda:",torch.cuda.is_available())
    if os.path.exists(pkl_name):
        cnn = torch.load(pkl_name).cuda()
    else:
        cnn = resnet50().cuda()

    print(cnn)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = torch.nn.CrossEntropyLoss()

    for i in range(80):
        # np_datas, np_labels = read_whole_datas(r'D:\code\PythonCode\PythonTest\pytorch\cifar_classes\imgs_train', 1000)
        np_datas = np.ones(shape=(100,3,64,64),dtype=np.float32)
        np_labels = np.ones(shape=(100),dtype=np.int64)
        torch_datas = torch.from_numpy(np_datas)
        torch_labels = torch.from_numpy(np_labels)

        torch_dataset = tdata.TensorDataset(torch_datas, torch_labels)

        loader = tdata.DataLoader(
            dataset=torch_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )

        for epoch in range(100):
            avg_loss = 0.0
            for step,(batch_x,batch_y) in enumerate(loader):
                output = cnn(batch_x.cuda())
                loss = loss_func(output.cpu(),batch_y)

                avg_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss /= step
            print("while(%d) epoch(%d) avg_loss(%0.6f)" %(i,epoch, avg_loss))
            if avg_loss < 0.01:
                break

            if epoch%10 == 0:
                torch.save(cnn,pkl_name)

        torch.save(cnn,pkl_name)

if __name__=='__main__':
    train_model()