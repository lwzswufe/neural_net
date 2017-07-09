# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
import tensorflow as tf
import numpy as np


x_dim = 20
y_dim = 3

x = tf.placeholder(tf.float32, [None, x_dim], name='x')    # value in the range of (0, 1)
y = tf.placeholder(tf.float32, [None, y_dim], name='y')

w = tf.Variable(tf.random_normal([x_dim, y_dim], stddev=0.1), dtype=tf.float32)
b = tf.Variable(tf.random_normal([y_dim], stddev=0.1), dtype=tf.float32)
y_ = tf.nn.tanh(tf.matmul(x, w) + b)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
train = tf.train.GradientDescentOptimizer(0.7).minimize(loss)
# train = tf.train.AdamOptimizer(0.1).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

x0 = np.float32(np.random.randn(11, x_dim))
y0 = np.float32(np.random.rand(11, y_dim))
pre_loss = 0

for step in range(100):
	loss_, w_, y__, _ = sess.run([loss, w, y_, train], {x: x0, y: y0})
	# 需要输出train才会执行优化
	print(pre_loss - loss_)
	pre_loss = loss_
	# print(y__[0])
	# print(np.sum(loss_))
