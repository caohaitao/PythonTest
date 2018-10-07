__author__ = 'ck_ch'
from torch.autograd import Variable
import torch.onnx
import torchvision
import torch

dummy_init = Variable(torch.randn(2936,1,28,28))
model = torch.load('chinese_character_2936.pkl')
torch.onnx.export(model,dummy_init,"chinese_character_2936.proto",verbose=True)