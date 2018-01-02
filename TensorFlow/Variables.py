import tensorflow as tf
# -*- coding: utf-8 -*-

#申请一个变量
state = tf.Variable(0,name='counter')

#定义一个常量one
one = tf.constant(1)

#定义加法步骤
new_value = tf.add(state,one)

#将state更新成new_value
update = tf.assign(state,new_value)

#如果定义了Variable，一定要initialize
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))