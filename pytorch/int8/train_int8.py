import torch
import torch.nn as nn
import numpy as np
import os
import sys
import torch.utils.data as tdata
import torch.nn.modules.conv as tconv
from torch.nn import functional as F
import torch.nn.modules.utils as ut

pkl_name = "int8.pkl"
onnx_name = "int8.onnx"

weight_scale = 243.5
input_scale = 1.28

EPOCH = 1000              # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # 学习率

def _rebuild_parameter(data, requires_grad, backward_hooks):
    param = torch.nn.Parameter(data, requires_grad)
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    param._backward_hooks = backward_hooks

    return param

def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
    tensor.requires_grad = requires_grad
    tensor._backward_hooks = backward_hooks
    return tensor

torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
torch._utils._rebuild_parameter = _rebuild_parameter

class MyConv2d(tconv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = ut._pair(kernel_size)
        stride = ut._pair(stride)
        padding = ut._pair(padding)
        dilation = ut._pair(dilation)
        super(MyConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, ut._pair(0), groups, bias)


    def forward(self, input):
        if self.training:
            weight = self.weight*weight_scale
            weight = torch.round(weight)
            weight = torch.clamp(weight,-128,127)
            return F.conv2d(input, weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)



class int8_cnn(nn.Module):
    def __init__(self):
        super(int8_cnn,self).__init__()

        self.conv1 = MyConv2d(1,4,3,1,1)

        self.fc = nn.Linear(64,1)


    def forward(self, x):
        if self.training:
            x = x * input_scale
            x = torch.round(x)
            x = torch.clamp(x, -128, 127)

        out = self.conv1(x)
        if self.training:
            out = out/(input_scale*weight_scale)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out


def get_train_data(num):
    res = np.random.random((num,16)).astype(np.float32)*100
    labels = np.sum(res,axis=1)
    res = res.reshape((num,1,4,4))
    labels = labels.reshape((num,1))
    return res,labels

def save_train_data():
    datas,labels = get_train_data(1000)
    for i in range(1000):
        file_name = format("inputs\\%d"%i)
        f = open(file_name,'wb+')
        f.write(datas[i].tobytes())
        f.close()

def train_model():
    if os.path.exists(pkl_name):
        cnn = torch.load(pkl_name)
    else:
        cnn = int8_cnn()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

    loss_func = torch.nn.MSELoss()

    datas,labels = get_train_data(1000)

    torch_datas = torch.from_numpy(datas)
    torch_labels = torch.from_numpy(labels)

    torch_dataset = tdata.TensorDataset(torch_datas,torch_labels)

    loader = tdata.DataLoader(
        dataset = torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    for epoch in range(50):
        avg_los = 0.0

        for step,(x,y) in enumerate(loader):
            output = cnn(x)
            loss = loss_func(output,y)
            avg_los += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_los /= step
        print("epoch(%d) avg_loss(%0.6f)" %
              (epoch, avg_los))

        if avg_los<0.001:
            break

    torch.save(cnn,pkl_name)

def export_model():
    cnn = torch.load(pkl_name).eval()

    dummp_input = torch.autograd.Variable(torch.randn(1, 1, 4, 4))
    torch.onnx.export(cnn, dummp_input, onnx_name, verbose=True)

def print_params():
    cnn = torch.load(pkl_name).eval()
    params = cnn.state_dict()
    for k,v in params.items():
        print(k,v.size())
        print(v.cpu())

if __name__ == '__main__':
    #train_model()
    #export_model()
    #save_train_data()
    print_params()