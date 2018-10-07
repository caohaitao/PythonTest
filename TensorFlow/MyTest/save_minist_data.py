from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import sys

# x=np.array([[1,2,3],[9,8,7],[6,5,4]])
# y=np.array([[1,2,3],[9,8,7],[6,5,4]])
# print("x shape is")
# print(x.shape)
# print("y shape is")
# print(y.shape)
# for i in x.shape:
#     print(x[i])

def get_y_num(batch_ys):
    i=0
    for a in batch_ys:
        if a != 0:
            break
        else:
            i = i+1
    return i

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_xs, batch_ys = mnist.train.next_batch(1)
print("batch_xs shape is")
print(batch_xs.shape)
print("batch_ys shape is")
print(batch_ys.shape)
i=0
for a in batch_xs[0]:
    if i%28 == 0:
        sys.stdout.write('\n')
    if a== 0:
        #print(0),
        sys.stdout.write(str(0))
    else:
        sys.stdout.write(str(1))
    sys.stdout.write(" ")
    i=i+1
sys.stdout.write('\n')
print(batch_ys[0])



print("y num is %d" % get_y_num(batch_ys[0]))

