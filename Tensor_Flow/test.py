# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
import tensorflow as tf
from Tensor_Flow import MyDataSet
import numpy as np

x_dim = 20
h_dim = 5

X0 = np.random.random([100, x_dim])
Y0 = np.transpose(np.mean(np.power(X0, 2), axis=1) + np.random.random([1, 100]) * 0.001)
mn = MyDataSet.DataSet(X0, Y0)
# Build a graph.
x = tf.placeholder(tf.float32, [None, x_dim])
W1 = tf.Variable(tf.ones([x_dim, h_dim]))
b1 = tf.Variable(tf.zeros([h_dim]))

W2 = tf.Variable(tf.ones([h_dim, 1]))
b2 = tf.Variable(tf.zeros([1]))

tmp = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.sigmoid(tf.matmul(tmp, W2) + b2)
y_ = tf.placeholder(tf.float32, [None, 1])
# dW = tf.Variable(tf.ones([784, 10]) * 0.1)
# Launch the graph in a session.
# W = W + dW
# W = tf.assign_add(W, dW)
sess = tf.Session()
cross_entropy = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run(session=sess)

flag = 0
for i in range(1000):
    batch_xs, batch_ys = mn.next_batch(5)
    W_, ts, loss, y0, y1 = sess.run(fetches=[W1, train_step, cross_entropy, y, y_], feed_dict={x: batch_xs, y_: batch_ys})
    flag = int(flag * 1.2)
    if i >= flag:
        print('W_00:', W_[0][0], 'loss:', loss)

batch_xs, batch_ys = mn.next_batch(5)
# 提取变量
print('y value:')
print(y.eval(session=sess, feed_dict={x: batch_xs, y_: batch_ys}))
