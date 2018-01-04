# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data读取测试数据，测试数据为很多张28x28的图片，上面是手写字，为0-9
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#添加层
def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

#计算当前模型的预测准确率，第一个参数为正确的输入图像，第二个参数为对应的标记
def compute_accuracy(v_xs, v_ys):
    global prediction
    #通过输入计算输出
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    #argmax函数为取数组中哪一个维度的最大值，当前为从计算结果中取最大值所在的index，相当于取出来是哪个数字，和
    #真实的数字做对比，得到一个真值矩阵
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    #cast为类型转换，reduce_mean为求平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #将真值表带入，求均值，其实就是求算对的概率
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
prediction = add_layer(xs, 784, 10,  activation_function=tf.nn.softmax)

#损失函数使用交叉熵
# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))
# loss 使用梯度下降优化损失函数
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

#运行测试数据
for i in range(1000):
    #每次随机取100张图片
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))

