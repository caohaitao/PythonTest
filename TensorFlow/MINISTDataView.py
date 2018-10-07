# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# number 1 to 10 data

#
# a = np.ndarray(shape=(2,3), dtype=int, buffer=np.array([1,2,3,4,5,6,7]), offset=0, order="C")
# print(a[0])

file_name_map={0:0}

def SaveOnData(b,c):
    number = 0
    for n in c:
        if n != 0:
            break
        else:
            number += 1

    if number in file_name_map.keys():
        file_name_map[number] += 1
    else:
        file_name_map[number] = 0
    # file_path=""
    file_path = format("d:\\vds\\pic_%d_%d.txt" % (number, file_name_map[number]))

    f = open(file_path, 'wb')
    i = 0
    width = 28
    height = 28
    bytes = width.to_bytes(4, byteorder='little')
    f.write(bytes)
    bytes = height.to_bytes(4, byteorder='little')
    f.write(bytes)
    for bx in b:
        iv = int(bx * 255)
        bytes = iv.to_bytes(1, byteorder='little')
        f.write(bytes)
        # f.write(",")
    f.close()

if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    batch_xs, batch_ys = mnist.train.next_batch(40)
    for index in range(40):
        bo = batch_xs[index]
        co = batch_ys[index]
        SaveOnData(bo,co)
