import torch
from torch import nn
from torch.autograd import Variable
import torch.onnx
import torchvision

#class Rnn(nn.Module):
class Rnn(torch.jit.ScriptModule):
    def __init__(self,INPUT_SIZE):
        super(Rnn,self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        self.out = nn.Linear(32,1)

    def forward(self, x,h_state):
        r_out,h_state = self.rnn(x,h_state)

        outs = []
        for time in range(r_out.size(1)):
            linear_input = r_out[:,time,:]
            outs.append(self.out(linear_input))
        return torch.stack(outs,dim=1),h_state

    def init_hidden(self,batch_size):
        weight = next(self.parameters()).data
        return Variable(weight.new(1, batch_size, 32).zero_())
