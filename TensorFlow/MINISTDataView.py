# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# number 1 to 10 data
print("begin")
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_xs, batch_ys = mnist.train.next_batch(1)
b = batch_xs[0]
c = batch_ys[0]
print("end")
#
# a = np.ndarray(shape=(2,3), dtype=int, buffer=np.array([1,2,3,4,5,6,7]), offset=0, order="C")
# print(a[0])

f = open("d:\\vd.txt",'wb')
i=0
for bx in b:
    # if i%28==0:
    #     f.write("\n")
    # if bx>0.5:
    #     f.write("1 ")
    # else:
    #     f.write("0 ")
    # i=i+1
    iv = int(bx*255)
    f.write(str(iv).encode())
    # f.write(",")
f.close()