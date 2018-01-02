# -*- coding: utf-8 -*-
#通过tensorflow自动优化线性函数y=ax+b，找到a和b
import tensorflow as tf
import numpy as np

#初始化测试数据
x_data = np.random.rand(100).astype(np.float32);
y_data = x_data*0.1+0.3

#构造参数，参数从-1-1选取
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
#初始化偏置项，全部为0
biases = tf.Variable(tf.zeros([1]))

#目标函数
y = Weights*x_data+biases

#定义损失函数
loss = tf.reduce_mean(tf.square(y-y_data))

#定义梯度学习速率
optimizer = tf.train.GradientDescentOptimizer(0.5)
#构造学习目标，最小化损失函数
train = optimizer.minimize(loss)

#初始化之前定义的变量
init = tf.global_variables_initializer()

#创建回话session
sess = tf.Session()
sess.run(init)

#循环执行训练，一步步的优化参数
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        aaa = sess.run(Weights)
        print(step,aaa,sess.run(biases))