# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from matplotlib import *
import pylab as plt

#参数：输入值，输入大小，输出大小，激励函数
def add_layer(inputs,in_size,out_size,activation_function=None):
    #初始化Weights为一个in_size行out_size列的随机变量矩阵
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    #偏置为0.1的矩阵
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    #未经过激活的值
    Wx_plus_b = tf.matmul(inputs,Weights)+biases

    #如果没有设置激活函数，则返回计算值
    if activation_function is None:
        outputs = Wx_plus_b
        #如果设置了激活函数，则返回经过激活的值
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#产生一个一维数组，数组从-1-1均值分布，总共300个
x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
#从高斯分布里面取数据，数据的均值为0，标准差为0.05
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
#y=x^2-0.5+noise
y_data = np.square(x_data)-0.5+noise

#创建两个占位符
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

#构造第一层layer，有一个输入，10个输出
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)

#增加一层layer，为输出层的layer，有10个输入，1个输出
prediction = add_layer(l1,10,1,activation_function=None)

#损失函数为 average(sum(y-y')^2),行求平均，当第二个参数为0时，对列求平均，如果不指定则是对所有元素求平均
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
#梯度下降训练，训练速率为0.1，对损失函数求最小
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#创建绘图环境
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

#构造session
sess = tf.Session()
sess.run(init)


#循环训练，输入为x_data，输出为y_data
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:
        #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value=sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(1)

