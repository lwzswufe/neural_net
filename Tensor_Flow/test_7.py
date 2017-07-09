# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

# This shows how to save/restore your model (trained variables).
# To see how it works, please stop this program during training and resart.
# This network is the same as 3_net.py


def init_weights(shape):
	return tf.Variable(shape, tf.float32)


def init_w(shape):
	return np.float32(np.random.randn(shape[0], shape[1]))


def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
	# this network is the same as the previous one except with an extra hidden layer + dropout
	X = tf.nn.dropout(X, p_keep_input)
	h = tf.nn.relu(tf.matmul(X, w_h))

	h = tf.nn.dropout(h, p_keep_hidden)
	h2 = tf.nn.relu(tf.matmul(h, w_h2))

	h2 = tf.nn.dropout(h2, p_keep_hidden)

	return tf.matmul(h2, w_o)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

w_h = init_weights([784, 625])
w_h2 = init_weights([625, 625])
w_o = init_weights([625, 10])

p_keep_input = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)

# py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)
X1 = tf.nn.dropout(X, p_keep_input)
h0 = tf.nn.relu(tf.matmul(X1, w_h))

h1 = tf.nn.dropout(h0, p_keep_hidden)
h2 = tf.nn.relu(tf.matmul(h1, w_h2))

h3 = tf.nn.dropout(h2, p_keep_hidden)
py_x = tf.matmul(h3, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
	os.makedirs(ckpt_dir)

global_step = tf.Variable(0, name='global_step', trainable=False)

# Call this after declaring all tf.Variables.
saver = tf.train.Saver()

# This variable won't be stored, since it is declared after tf.train.Saver()
non_storable_variable = tf.Variable(777)

# Launch the graph in a session
with tf.Session() as sess:
	# you need to initialize all variables
	tf.global_variables_initializer().run()

	w_h_init = init_w([784, 625])
	w_h2_init = init_w([625, 625])
	w_o_init = init_w([625, 10])

	ckpt = tf.train.get_checkpoint_state(ckpt_dir)
	start = 0
	print("Start from:", start)

	for i in range(start, 100):
		tr_x, tr_y = mnist.train.next_batch(128)
		t, w = sess.run([train_op, w_h], feed_dict={X: tr_x, Y: tr_y,
			                              p_keep_input: 0.8, p_keep_hidden: 0.5})
		print(np.mean(w_h_init[0]))

		global_step.assign(i).eval()  # set and update(eval) global_step with index, i
		# saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)
		# feed_dict = {X: teX, p_keep_input: 1.0, p_keep_hidden: 1.0}
		# print(i, np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict)))
