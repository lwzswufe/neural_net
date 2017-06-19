# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
import tensorflow as tf
from Tensor_Flow import MyDataSet
import numpy as np


mn = MyDataSet.DataSet(np.zeros([100, 28, 28, 1]), np.ones([100, 10]))
# Build a graph.
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 10])
# Launch the graph in a session.
sess = tf.Session()
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run(session=sess)
for _ in range(10):
    batch_xs, batch_ys = mn.next_batch(5)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

batch_xs, batch_ys = mn.next_batch(5)
# 提取变量
print(y.eval(session=sess, feed_dict={x: batch_xs, y_: batch_ys}))
