from PIL import Image
import torchvision
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tdata
import random

GPUID = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID
pkl_name = "class_my.pkl"

EPOCH = 1000              # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 10
LR = 0.0001              # 学习率


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

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']
class cnn_class(nn.Module):
    def __init__(self):
        super(cnn_class,self).__init__()

        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels,v,3,padding=1)
                layers += [conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
                in_channels = v

        self.features = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(2048, 500),
            nn.ReLU(),
            nn.Linear(500,10)
        )

    def forward(self,x):
        convs = self.features(x)
        convs = convs.view(convs.size(0),-1)
        out = self.fc(convs)
        return out

def train_model():
    if os.path.exists(pkl_name):
        cnn = torch.load(pkl_name).cuda().train()
    else:
        cnn = cnn_class().cuda().train()

    print(cnn)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = torch.nn.CrossEntropyLoss()

    for i in range(80):
        np_datas, np_labels = read_whole_datas(r'D:\code\PythonCode\PythonTest\pytorch\cifar_classes\imgs_train', 1000)
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
            if avg_loss < 0.001:
                break

            if epoch%10 == 0:
                torch.save(cnn,pkl_name)

        torch.save(cnn,pkl_name)

if __name__=='__main__':
    train_model()