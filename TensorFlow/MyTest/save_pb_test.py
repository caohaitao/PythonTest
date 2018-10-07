from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import os
import glob
from skimage import io, transform
from tensorflow.python.framework import graph_util
import collections
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
w = 28
h = 28
c = 1

def build_network(height, width, channel):
    x = tf.placeholder(tf.float32, shape=[None, height,width, channel], name='input')
    y_ = tf.placeholder(tf.int64, shape=[None,], name='y_')

    def weight_variable(shape, name="weights"):
        initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)*0.001
        return tf.Variable(initial, name=name)

    def bias_variable(shape, name="biases"):
        initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(input, w):
        return tf.nn.conv2d(input, w, [1, 1, 1, 1], padding='SAME')

    def pool_max(input):
        return tf.nn.max_pool(input,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

    def fc(input, w, b):
        return tf.matmul(input, w) + b

    # conv1
    with tf.name_scope('conv1_1') as scope:
        kernel = weight_variable([3, 3, 1, 32])
        biases = bias_variable([32])
        output_conv1_1 = tf.nn.relu(conv2d(x, kernel) + biases, name=scope)

    pool1 = pool_max(output_conv1_1) # [n,14,14,32]

    # conv2
    with tf.name_scope('conv1_2') as scope:
        kernel = weight_variable([3, 3, 32, 64])
        biases = bias_variable([64])
        output_conv1_2 = tf.nn.relu(conv2d(pool1, kernel) + biases, name=scope)

    pool2 = pool_max(output_conv1_2) # [n,7,7,64]

    #fc1
    with tf.name_scope('fc1') as scope:
        shape = int(np.prod(pool2.get_shape()[1:]))
        kernel = weight_variable([shape, 512])
        biases = bias_variable([512])
        pool2_flat = tf.reshape(pool2, [-1, shape])
        output_fc1 = tf.nn.relu(fc(pool2_flat, kernel, biases), name=scope)

    #fc2
    with tf.name_scope('fc2') as scope:
        kernel = weight_variable([512, 10])
        biases = bias_variable([10])
        output_fc2 = fc(output_fc1, kernel, biases)

    y = tf.nn.softmax(output_fc2, name="softmax")

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_))
    optimize = tf.train.AdamOptimizer(learning_rate=1e-1).minimize(cost)

    prediction_labels = tf.argmax(y, axis=1, name="output")

    correct_prediction = tf.equal(prediction_labels, y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    correct_times_in_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

    return dict(
        x=x,
        y_=y_,
        optimize=optimize,
        correct_prediction=correct_prediction,
        correct_times_in_batch=correct_times_in_batch,
        accuracy=accuracy,
        cost=cost,
    )


def train_network(graph, batch_size, num_epochs, pb_file_path):
    init = tf.global_variables_initializer()

    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        # 验证之前是否已经保存了检查点文件
        ckpt = tf.train.get_checkpoint_state('./output/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for epoch_index in range(num_epochs):
            for i in range(1000):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                feed={
                    graph['x']: np.reshape(batch_xs, (-1, h, w, c)),
                    graph['y_']: batch_ys}
                sess.run([graph['optimize']], feed_dict=feed)
                if i%100==0:
                    print('step',i,'acc',sess.run(graph['accuracy'],feed),'loss',sess.run(graph['cost'],feed))

                saver.save(sess,'./output/model.ckpt',global_step=i)

            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,['output'])
            with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())


def main():
    batch_size = 128
    num_epochs = 2

    pb_file_path = "mnist.pb"

    g = build_network(height=h, width=w, channel=c)
    train_network(g, batch_size, num_epochs, pb_file_path)

main()