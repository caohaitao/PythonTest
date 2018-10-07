__author__ = 'ck_ch'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MINIST_data",one_hot=True)
print(mnist.train.images.shape,mnist.train.labels.shape)