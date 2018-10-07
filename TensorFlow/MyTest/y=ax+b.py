# -*- coding: utf-8 -*-
__author__ = 'ck_ch'
#加载tensorflow和numpy
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util

#定义输入数据，现在模拟函数为y=0.1*x+0.3
x_data = np.random.rand(100).astype(np.float32)
#y作为真实的输入值
y_data = x_data*0.1 + 0.3

#产生一个变量Weights，这里只是定义变量的类型和取值范围
#random_uniform 产生一个1维的随机值，这个值从-1~1之间取值
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
#定义变量为0
biases = tf.Variable(tf.zeros([1]))

#构造计算函数
y = Weights * x_data + biases

#误差计算，采用均方差 （△y）²，reduce_mean为取平均的意思
loss = tf.reduce_mean(tf.square(y-y_data))

#使用梯度下降法构建优化器
optimizer = tf.train.GradientDescentOptimizer(0.5)
#训练器要使得误差减小
train = optimizer.minimize(loss)

#初始化操作
init = tf.global_variables_initializer()

#创建session会话
sess = tf.Session()
#使用session执行init初始化
sess.run(init)

#循环200次
for step in range(201):
    #使用会话训练网络
    sess.run(train)
    print(step,sess.run(Weights),sess.run(biases))

graph_def = tf.get_default_graph().as_graph_def()
output_graph_def = graph_util.convert_variables_to_constants(sess,graph_def,['add'])

with tf.gfile.GFile("Model/ax_plus_b.pb",'wb') as f:
    f.write(output_graph_def.SerializeToString())
