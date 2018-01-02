# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

speed = 0.01
def sigmoid(x):
    return 1/(1+np.exp(-x))

xs=[
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
]
results=[1,0,0,1]

ws = np.random.randn(3,3)/1.22

aas=[0,0,0]
das= [0,0,0]
# a = ws[0]*xs[0].T
# print(a)

def Multy(ins,ins2):
    res = 0.0
    for i in range(3):
        res += ins[i]*ins2[i]
    return res

for i in range(1000000):
    for x, result in zip(xs, results):
        aas[0] = sigmoid(Multy(ws[0],x))
        aas[1] = sigmoid(Multy(ws[1],x))
        as_layer2 = [aas[0], aas[1], 1.0]
        aas[2] = sigmoid(Multy(as_layer2,ws[2]))

        das[2] = aas[2] * (1 - aas[2]) * (result - aas[2])
        das[1] = aas[1] * (1 - aas[1]) * ws[2][1] * das[2]
        das[0] = aas[0] * (1 - aas[0]) * ws[2][0] * das[2]

        ws[2][0] += aas[2] * (1 - aas[2]) * speed * das[2] * aas[0]
        ws[2][1] += aas[2] * (1 - aas[2]) * speed * aas[1] * das[2]
        ws[2][2] += aas[2] * (1 - aas[2]) * speed * das[2]

        ws[0][0] += aas[0] * (1 - aas[0]) * speed * x[0] * das[0]
        ws[0][1] += aas[0] * (1 - aas[0]) * speed * x[1] * das[0]
        ws[0][2] += aas[0] * (1 - aas[0]) * speed * x[2] * das[0]

        ws[1][0] += aas[1] * (1 - aas[1]) * speed * x[0] * das[1]
        ws[1][1] += aas[1] * (1 - aas[1]) * speed * x[1] * das[1]
        ws[1][2] += aas[1] * (1 - aas[1]) * speed * x[2] * das[1]


outs=[]
for x,result in zip(xs,results):
    z = float(ws[0]*np.mat(x).T)
    aas[0] = sigmoid(z)
    aas[1] = sigmoid(float(ws[1]*np.mat(x).T))
    as_layer2 = np.mat([aas[0],aas[1],1.0])
    wst = np.mat(ws[2]).T
    z=as_layer2 * wst
    z=float(z)
    aas[2] = sigmoid(z)
    outs.append(aas[2])

print(outs)