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
	print('init_w')
	return tf.Variable(tf.random_normal(shape, stddev=0.01))


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

w_h = tf.Variable(initial_value=tf.random_normal([784, 625], stddev=0.01), dtype=tf.float32)
w_h2 = tf.Variable(initial_value=tf.random_normal([625, 625], stddev=0.01), dtype=tf.float32)
w_o = tf.Variable(initial_value=tf.random_normal([625, 10], stddev=0.01), dtype=tf.float32)

print(p_keep_hidden)

diag_w2 = tf.diag_part(w_h2)

X1 = tf.nn.dropout(X, p_keep_input)
h0 = tf.nn.relu(tf.matmul(X1, w_h))

h1 = tf.nn.dropout(h0, p_keep_hidden)
h2 = tf.nn.relu(tf.matmul(h1, w_h2))

h3 = tf.nn.dropout(h2, p_keep_hidden)
py_x = tf.matmul(h3, w_o)

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
cost = tf.reduce_mean(tf.square(py_x - Y))
# train_op = tf.train.AdamOptimizer(0.01).minimize(cost)
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
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
	tf.global_variables_initializer().run(session=sess)

	start = 0
	print("Start from:", start)
	z = 0.0

	for i in range(start, 100):
		tr_x, tr_y = mnist.train.next_batch(128)
		t, w, loss, d= sess.run([train_op, w_h, cost, diag_w2], feed_dict={X: tr_x, Y: tr_y,
			                              p_keep_input: 0.9, p_keep_hidden: 0.9})
		print('step:', str(i), ' ', z - np.mean(w[0]), ' loss_', str(loss))
		z = np.mean(w[0])

		global_step.assign(i).eval()  # set and update(eval) global_step with index, i
		# saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)

