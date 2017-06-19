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


# 定义权重函数
def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))


# 定义模型
def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
	# this network is the same as the previous one except with an extra hidden layer + dropout
	# 第一个全连接层
	X = tf.nn.dropout(X, p_keep_input)
	h = tf.nn.relu(tf.matmul(X, w_h))

	h = tf.nn.dropout(h, p_keep_hidden)
	# 第二个全连接层
	h2 = tf.nn.relu(tf.matmul(h, w_h2))

	h2 = tf.nn.dropout(h2, p_keep_hidden)
	# 返回预测值
	return tf.matmul(h2, w_o)

# 获取数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

assert trX.shape[1] == teX.shape[1]
x_dim = trX.shape[1]
h_dim = 625
y_dim = 10
batch_size = 128

X = tf.placeholder("float", [None, x_dim])
Y = tf.placeholder("float", [None, y_dim])

w_h = init_weights([x_dim, h_dim])
w_h2 = init_weights([h_dim, h_dim])
w_o = init_weights([h_dim, y_dim])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

# 定义损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# 定义存储路径
ckpt_dir = "D:\\Cache\\ckpt_dir"
print('储存路径是{}'.format(ckpt_dir))
if not os.path.exists(ckpt_dir):
	os.makedirs(ckpt_dir)

# 计数器变量
global_step = tf.Variable(0, name='global_step', trainable=False)

# Call this after declaring all tf.Variables.
# 申明完所有变量之后调用tf.train.Saver
saver = tf.train.Saver()

# This variable won't be stored, since it is declared after tf.train.Saver()
# 位于tf.train.Saver之后的变量不会被储存
non_storable_variable = tf.Variable(777)

# Launch the graph in a session
with tf.Session() as sess:
	# you need to initialize all variables
	tf.global_variables_initializer().run()

	ckpt = tf.train.get_checkpoint_state(ckpt_dir)
	if ckpt and ckpt.model_checkpoint_path:
		print(ckpt.model_checkpoint_path)
		saver.restore(sess, ckpt.model_checkpoint_path)  # restore all variables

	start = global_step.eval()  # get last global_step
	print("Start from:", start)

	for i in range(start, 100):
		print('iter_{}'.format(str(i)))
		for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
			sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})
		# 更新计数器
		global_step.assign(i).eval()  # set and update(eval) global_step with index, i
		# 储存模型
		saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)
		print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX,
                                                         p_keep_input: 1.0,
                                                         p_keep_hidden: 1.0})))