# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
"""
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import math
from Tensor_Flow import load_data, MyDataSet
import os
import pickle


class Param(object):
	def __init__(self, dataset_path='D:\\Code\\Code\\neural_net\\Tensor_Flow\\MNIST_data',
	batch_size=64, learning_rate=0.002, show_img_num=5, train_times=1000, h_dim=[64, 32, 16]):
		self.batch_size = batch_size
		self.lr = learning_rate         # learning rate
		self.show_img_num = show_img_num     # 可视化显示图片数量
		self.train_times = train_times
		self.h_dim_1 = h_dim[0]
		self.h_dim_2 = h_dim[1]
		self.h_dim_3 = h_dim[2]
		self.x_dim = 0
		self.dataset_path = dataset_path
		self.length = 0
		self.width = 0


class AutoEncoder(object):
	def __init__(self, param):
		self.param = param
		self.sess = ''

	def input_data(self, data=None):
		# Mnist digits
		# mnist = input_data.read_data_sets('./mnist', one_hot=False)     # use not one-hotted target data
		if data is None:
			self.data = input_data.read_data_sets(param.dataset_path, one_hot=True)
		else:
			self.data = data

		# plot one example
		print('images shape: ', self.data.train.images.shape)     # (55000, 28 * 28)
		print('lebal  shape: ', self.data.train.labels.shape)     # (55000, 10)
		self.param.x_dim = self.data.train.images.shape[1]
		self.param.width = math.sqrt(self.param.x_dim)
		self.param.length = math.sqrt(self.param.x_dim)
		if self.data.train.image_size == 784:
			plt.imshow(self.data.train.images[0].reshape((self.param.length, self.param.width)), cmap='gray')
			plt.title('%i' % np.argmax(self.data.train.labels[0]))
			plt.show()

	def train(self):
		tf.set_random_seed(1)
		# tf placeholder
		tf_x = tf.placeholder(tf.float32, [None, self.param.x_dim])    # value in the range of (0, 1)

		# dense(inputs, units, activation=None, use_bias=True, kernel_initializer=None, bias_initializer
		# encoder
		en0 = tf.layers.dense(tf_x, self.param.h_dim_1, tf.nn.tanh)
		en1 = tf.layers.dense(en0, self.param.h_dim_2, tf.nn.tanh)
		en2 = tf.layers.dense(en1, self.param.h_dim_3, tf.nn.tanh)
		encoded = tf.layers.dense(en2, 16)

		# decoder
		de0 = tf.layers.dense(encoded, self.param.h_dim_3, tf.nn.tanh)
		de1 = tf.layers.dense(de0, self.param.h_dim_2, tf.nn.tanh)
		de2 = tf.layers.dense(de1, self.param.h_dim_1, tf.nn.tanh)
		decoded = tf.layers.dense(de2, self.param.x_dim, tf.nn.sigmoid)

		loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded)
		train = tf.train.AdamOptimizer(self.param.lr).minimize(loss)

		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		self.sess = sess

		if self.data.train.image_size == 784:
			# initialize figure
			f, a = plt.subplots(2, self.param.show_img_num, figsize=(self.param.show_img_num, 2))
			plt.ion()   # continuously plot
			# original data (first row) for viewing
			view_data = self.data.test.images[:self.param.show_img_num]
			for i in range(self.param.show_img_num):
				a[0][i].imshow(np.reshape(view_data[i], (self.param.length, self.param.width)), cmap='gray')
				a[0][i].set_xticks(())
				a[0][i].set_yticks(())

		for step in range(self.param.train_times):
			b_x, b_y = self.data.train.next_batch(self.param.show_img_num)
			_, encoded_, decoded_, loss_ = sess.run([train, encoded, decoded, loss], {tf_x: b_x})

			if step % 100 == 0 and self.data.train.image_size == 784:     # plotting
				print('train loss: %.4f' % loss_)
				# plotting decoded image (second row)
				decoded_data = sess.run(decoded, {tf_x: view_data})
				for i in range(self.param.show_img_num):
					a[1][i].clear()
					a[1][i].imshow(np.reshape(decoded_data[i], (self.param.length, self.param.width)), cmap='gray')
					a[1][i].set_xticks(())
					a[1][i].set_yticks(())
				plt.draw()
				plt.pause(0.01)
			elif step % 100 == 0:
				print('%s step train loss: %.4f' % (str(step), loss_))

		if self.data.train.image_size == 784:
			plt.ioff()

		all_images, all_labels, all_codes, all_dates, all_profits = self.data.get_all()
		encodes = encoded.eval(session=self.sess, feed_dict={tf_x: all_images})
		with open('D:\\Cache\\encodes.pic', 'wb') as f:
			pickle.dump([encodes, all_codes, all_dates, all_profits], f)


def sample_generator(data, n_day=5):
	data = data[data[:, 7] > 0, :]
	length = data.shape[0]
	sample_length = 100
	interval = 10
	sample = list()
	label = list()
	date_s = list()
	profit_s = list()
	if length < sample_length:
		return [], [], [], []
	else:
		data[:, 5] = np.log(data[:, 5])
		st = length % interval
		ed = st + sample_length
		while ed <= length:
			k = data[st: ed, 5]
			assert k.shape[0] == sample_length
			sample.append(k - np.mean(k) + 0.5)
			label.append(np.ones([1, 1]))
			date_s.append(data[ed - 1, 0])
			nday_profit = data[min(ed - 1 + n_day, length - 1), 5] - data[ed - 1, 5]
			profit_s.append(nday_profit)
			st, ed = st + interval, ed + interval
	return sample, label, date_s, profit_s


if __name__ == '__main__':
	fn = 'D:\\Cache\\autoencoder.pic'
	if os.path.exists(fn) and False:
		c = MyDataSet.load_pick(fn)
	else:
		c = load_data.collect_data(sample_generator, limit=0)
		load_data.save(c, fn)
	param = Param(train_times=20000, learning_rate=0.002)
	ae = AutoEncoder(param)
	ae.input_data(data=c)
	ae.train()
