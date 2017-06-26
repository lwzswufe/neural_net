# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
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
	batch_size=64, learning_rate=0.002, show_img_num=5, train_times=1000, h_dim=500):
		self.batch_size = batch_size
		self.lr = learning_rate         # learning rate
		self.show_img_num = show_img_num     # 可视化显示图片数量
		self.train_times = train_times
		self.h_dim = h_dim
		self.x_dim = 0
		self.dataset_path = dataset_path
		self.length = 0
		self.width = 0


class rbm(object):
	def __init__(self, param=Param(), is_plot=True):
		self.param = param
		self.is_plot = is_plot
		self.sess = None
		self.cache = None
		self.w = None
		self.v_b = None
		self.h_b = None

	def input_data(self, data=None):
		# Mnist digits
		# mnist = input_data.read_data_sets('./mnist', one_hot=False)     # use not one-hotted target data
		if data is None:
			self.data = input_data.read_data_sets(self.param.dataset_path, one_hot=True)
		else:
			self.data = data

		# plot one example
		print('images shape: ', self.data.train.images.shape)     # (55000, 28 * 28)
		print('lebal  shape: ', self.data.train.labels.shape)     # (55000, 10)

		self.param.x_dim = self.data.train.images.shape[1]
		self.param.width = round(math.sqrt(self.param.x_dim))
		self.param.length = round(math.sqrt(self.param.x_dim))

	def train(self):
		tf.set_random_seed(1)
		# tf placeholder
		v0 = tf.placeholder("float", [None, self.param.x_dim])

		rbm_w = tf.placeholder("float", [self.param.x_dim, self.param.h_dim])
		rbm_vb = tf.placeholder("float", [self.param.x_dim])
		rbm_hb = tf.placeholder("float", [self.param.h_dim])

		h0 = sample_prob(tf.nn.sigmoid(tf.matmul(v0, rbm_w) + rbm_hb))
		v1 = sample_prob(tf.nn.sigmoid(tf.matmul(h0, tf.transpose(rbm_w)) + rbm_vb))
		h1 = tf.nn.sigmoid(tf.matmul(v1, rbm_w) + rbm_hb)

		"""tf.transpose(v0), h0 意为v0' * h0"""
		w_positive_grad = tf.matmul(tf.transpose(v0), h0)
		w_negative_grad = tf.matmul(tf.transpose(v1), h1)

		alpha = + self.param.lr
		d_w = alpha * (w_positive_grad - w_negative_grad) / self.param.batch_size
		d_vb = alpha * tf.reduce_mean(v0 - v1, 0)
		d_hb = alpha * tf.reduce_mean(h0 - h1, 0)

		update_w = rbm_w + d_w
		update_vb = rbm_vb + d_vb
		update_hb = rbm_hb + d_hb
		""" update = tf.assign(state,new_value)
		这个操作是：赋值操作。将new_value的值赋值给update变量。"""

		h_prop = tf.nn.sigmoid(tf.matmul(v0, rbm_w) + rbm_hb)
		h_sample = sample_prob(h_prop)
		v_prop = tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(rbm_w)) + rbm_vb)
		v_sample = sample_prob(v_prop)
		# v_sample = tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(rbm_w)) + rbm_vb)
		# v_sample = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(rbm_w)) + rbm_vb)

		err = v0 - v_sample
		err_sum = tf.reduce_mean(err * err)

		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		self.sess = sess

		old_w = np.zeros([self.param.x_dim, self.param.h_dim], np.float32)
		old_vb = np.zeros([self.param.x_dim], np.float32)
		old_hb = np.zeros([self.param.h_dim], np.float32)

		if self.is_plot:  # self.data.train.image_size == 784:
			# initialize figure
			f, a = plt.subplots(3, self.param.show_img_num, figsize=(self.param.show_img_num, 3))
			plt.ion()   # continuously plot
			# original data (first row) for viewing
			view_data = self.data.test.images[:self.param.show_img_num]
			for i in range(self.param.show_img_num):
				a[0][i].imshow(np.reshape(view_data[i], (self.param.length, self.param.width)), cmap='gray')
				a[0][i].set_xticks(())
				a[0][i].set_yticks(())

		plot_step = 1
		for step in range(self.param.train_times):
			x, _ = self.data.train.next_batch(self.param.batch_size)
			feed_dict = {v0: x, rbm_w: old_w, rbm_hb: old_hb, rbm_vb: old_vb}
			old_w, old_hb, old_vb, loss_ = sess.run([update_w, update_hb, update_vb, err_sum], feed_dict)
			# print(sum(old_w[1, :]))

			if step > plot_step and self.is_plot:      # plotting
				plot_step = plot_step * 1.2 + 10
				print('step %s train loss: %.4f sum_w %.4f' % (step, loss_, sum(old_w[1, :])))
				# plotting decoded image (second row)
				feed_dict = {v0: view_data, rbm_w: old_w, rbm_hb: old_hb, rbm_vb: old_vb}
				v_ = sess.run([v_sample, v_prop], feed_dict)
				for ii in [1, 2]:
					for i in range(self.param.show_img_num):
						a[ii][i].clear()
						a[ii][i].imshow(np.reshape(v_[ii-1][i], (self.param.length, self.param.width)), cmap='gray')
						a[ii][i].set_xticks(())
						a[ii][i].set_yticks(())
				plt.draw()
				plt.pause(0.01)
			elif step > plot_step:
				plot_step *= 1.5
				print('step %s train loss: %.4f sum_w %.4f' % (step, loss_, sum(old_w[1, :])))

		if self.is_plot:
			plt.ioff()
		self.w = old_w
		self.v_b = old_vb
		self.h_b = old_hb

		if False:
			feed_dict = {v0: self.data.train.images, rbm_w: old_w, rbm_hb: old_hb, rbm_vb: old_vb}
			h_, v_ = sess.run([h_sample, v_prop], feed_dict)
			return h_, v_
		# all_images, all_labels, all_codes, all_dates = self.data.get_all()
		# feed_dict = {v0: self.data.train.images, rbm_w: old_w, rbm_hb: old_hb, rbm_vb: old_vb}
		# encodes = encoded.eval(session=self.sess, feed_dict={tf_v0: all_images})
		# with open('D:\\Cache\\encodes.pic', 'wb') as f:
		# 	pickle.dump([encodes, all_codes, all_dates], f)

	def get_next_layer_dataset(self):
		h_, h = self.get_hidden(data=self.data.train.images)
		train = MyDataSet.DataSet(images=h, labels=self.data.train.labels)

		h_, h = self.get_hidden(data=self.data.validation.images)
		validation = MyDataSet.DataSet(images=h, labels=self.data.validation.labels)

		h_, h = self.get_hidden(data=self.data.test.images)
		test = MyDataSet.DataSet(images=h, labels=self.data.test.labels)

		z = MyDataSet.Datasets(train=train, validation=validation, test=test)
		return z

	def get_hidden(self, data):
		if data is None:
			data = self.data.train.images

		v0 = tf.placeholder("float", [None, self.param.x_dim])
		rbm_w = tf.placeholder("float", [self.param.x_dim, self.param.h_dim])
		rbm_hb = tf.placeholder("float", [self.param.h_dim])
		h_p = tf.nn.sigmoid(tf.matmul(v0, rbm_w) + rbm_hb)
		h_s = sample_prob(h_p)

		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		feed_dict = {v0: data, rbm_w: self.w, rbm_hb: self.h_b}
		h_sample, h_prop = sess.run([h_s, h_p], feed_dict)
		# return h_sample, h_prop
		return h_prop, h_prop

	def get_visible(self, data):
		h0 = tf.placeholder("float", [None, self.param.h_dim])
		rbm_w = tf.placeholder("float", [self.param.x_dim, self.param.h_dim])
		rbm_vb = tf.placeholder("float", [self.param.x_dim])
		v_p = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(rbm_w)) + rbm_vb)
		v_s = sample_prob(v_p)

		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		feed_dict = {h0: data, rbm_w: self.w, rbm_vb: self.v_b}
		v_sample, v_prop = sess.run([v_s, v_p], feed_dict)
		# return v_sample, v_prop
		return v_prop, v_prop


def sample_prob(prob):
	"""Do sampling with the given probability (you can use binomial in Theano)
	依概率将神经元设置为0或1"""
	return tf.nn.relu(tf.sign(prob - tf.random_uniform(tf.shape(prob))))


def sample_generator(data):
	data = data[data[:, 7] > 0, :]
	length = data.shape[0]
	sample_length = 100
	interval = 10
	sample = list()
	label = list()
	date_s = list()
	if length < sample_length:
		return [], [], []
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
			st, ed = st + interval, ed + interval
	return sample, label, date_s


if __name__ == '__main__':
	param = Param(train_times=20000, learning_rate=0.002, h_dim=100)
	ae = rbm(param, is_plot=True)
	ae.input_data()
	ae.train()
