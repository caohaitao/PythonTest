import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs

x_data = np.ndarray(shape=(4,2), dtype=float,
                    buffer=np.array([0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0]), offset=0, order="C")
y_data = np.ndarray(shape=(4,1), dtype=float,
                    buffer=np.array([0.0,1.0,1.0,0.0]),offset=0,order="C")

with tf.name_scope('inputs_my'):
    xs = tf.placeholder(tf.float32,[None,2],name='x_input_my')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input_my')

l1 = add_layer(xs,2,2,n_layer=1,activation_function=tf.nn.sigmoid)

prediction = add_layer(l1,2,1,n_layer=2,activation_function=tf.nn.sigmoid)

with tf.name_scope('loss_my'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    tf.summary.scalar('loss_my',loss)
with tf.name_scope('train_my'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)




sess = tf.Session()
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("logs/",sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%100==0:
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)
        #prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        #print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

#print(sess.run(prediction, feed_dict={xs: x_data, ys: y_data}))