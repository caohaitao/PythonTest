# -*- coding: utf-8 -*-
__author__ = 'ck_ch'
#加载tensorflow和numpy
import tensorflow as tf
import numpy as np

#构造x变量，值为1
x = tf.Variable(0,name='counter')
#构造b常量，值为1
b = tf.constant(1)

#构造运算过程，y=x+b
y = tf.add(x,b)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        #给x付新的值
        x.load(i,sess)
        #运算y
        sess.run(y)
        #打印y
        print(sess.run(y))
