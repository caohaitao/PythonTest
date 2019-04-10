import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import sin_cos_rnn
import os

TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

pkl_name = 'sin_cos_rnn.pkl'

def train_rnn():
    model = sin_cos_rnn.Rnn(INPUT_SIZE)
    print(model)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    h_state = None

    for step in range(300):
        start, end = step * np.pi, (step + 1) * np.pi

        steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
        x_np = np.sin(steps)
        y_np = np.cos(steps)

        x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
        y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
        print('x=',x,'y=',y)

        prediction, h_state = model(x, h_state)
        h_state = h_state.data

        loss = loss_func(prediction, y)
        # if loss < 0.02:
        #     break
        print('step(%d) loss=' % (step), loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.plot(steps,y_np.flatten(),'r-')
    plt.plot(steps,prediction.data.numpy().flatten(),'b-')
    plt.show()

    #example = torch.rand(1,10,1);
    #traced_script_module = torch.jit.trace(model,example)

    #traced_script_module.save(pkl_name)

    #torch.save(model, pkl_name)
    t = torch.jit.trace(model,(torch.rand(1,10,1),torch.zeros(1,1,32)))
    torch.jit.save(t,pkl_name)

    # dummy_init = torch.autograd.Variable(torch.randn(1, 10,1))
    # dummy_state = torch.zeros(1, 1, 32)
    # torch.onnx.export(model, (dummy_init,dummy_state), "sin_cos.proto", verbose=True)

def show_rnn():
    model = torch.load(pkl_name)
    loss_func = nn.MSELoss()

    h_state = None
    for step in range(10):
        start, end = step * np.pi, (step + 1) * np.pi

        steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
        x_np = np.sin(steps)
        y_np = np.cos(steps)

        x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
        y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

        #print('x=', x, 'y=', y)
        prediction, h_state = model(x, h_state)
        h_state = h_state.data

        loss = loss_func(prediction, y)
        print('step=%d,los='%(step),loss)

        plt.plot(steps,y_np.flatten(),'r-')
        plt.plot(steps,prediction.data.numpy().flatten(),'b-')
        plt.show()

if __name__ == '__main__':
    if os.path.exists(pkl_name):
        show_rnn()
    else:
        train_rnn()