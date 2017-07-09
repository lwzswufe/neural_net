# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from Tensor_Flow import MyDataSet


class Param(object):
	def __init__(self, dataset_path='D:\\Code\\Code\\neural_net\\Tensor_Flow\\MNIST_data',
	             batch_size=200, lr_g=0.002, lr_d=0.002, show_img_num=9, train_times=1000,
	             fig_size=784, n_ideas=7, h_dim=10):
		self.batch_size = batch_size
		self.lr_d = lr_d         # learning rate
		self.lr_g = lr_g
		self.show_img_num = show_img_num     # 可视化显示图片数量
		self.train_times = train_times
		self.fig_size = fig_size
		self.fig_shape = (int(np.sqrt(fig_size)), int(np.sqrt(fig_size)))
		self.x_dim = 0
		self.h_dim = 10
		self.dataset_path = dataset_path
		self.length = 0
		self.width = 0
		self.n_ideas = n_ideas


class gan(object):
	def __init__(self, param=Param(), is_plot=False):
		self.param = param
		self.pp = self.get_ideas()
		self.data = ''
		self.coder = Codes()
		self.is_plot = is_plot
		pass

	def get_data(self):
		with open("D:\\Cache\\ckpt_dir\\encodes_mnist.pic", 'rb') as f:
			data = pickle.load(f)
			self.data = data[0]
			self.coder = Codes(data[1])
			self.param.x_dim = self.data.images.shape[1]
			self.param.h_dim = self.data.labels.shape[1]

	def get_ideas(self):     # painting from the famous artist (real target)
		fake_data = np.random.randn(self.param.batch_size, self.param.n_ideas)
		return fake_data

	def train_Gan(self):
		with tf.variable_scope('Generator'):
			G_in = tf.placeholder(tf.float32, [None, self.param.n_ideas])          # random ideas (could from normal distribution)
			G_l1 = tf.layers.dense(G_in, 8, tf.nn.sigmoid)
			G_out = tf.layers.dense(G_l1, self.param.x_dim, tf.nn.sigmoid)               # making a painting from these random ideas

		with tf.variable_scope('Discriminator'):
			real_art = tf.placeholder(tf.float32, [None, self.param.x_dim], name='real_in')   # receive art work from the famous artist

			D_l0 = tf.layers.dense(real_art, self.param.x_dim, tf.nn.relu, name='l')
			prob_artist0 = tf.layers.dense(D_l0, self.param.h_dim, tf.nn.sigmoid, name='out')              # probability that the art work is made by artist
			# reuse layers for generator
			D_l1 = tf.layers.dense(G_out, self.param.x_dim, tf.nn.relu, name='l', reuse=True)            # receive art work from a newbie like G
			prob_artist1 = tf.layers.dense(D_l1, self.param.h_dim, tf.nn.sigmoid, name='out', reuse=True)  # probability that the art work is made by artist

		y_label = tf.placeholder(tf.float32, [None, self.param.h_dim])

		# D0_loss = tf.diag_part(tf.matmul(prob_artist0, y_label, transpose_b=True))
		D0_loss = tf.reduce_mean(tf.square(prob_artist0 - y_label))
		D_loss = - tf.reduce_mean(-tf.log(D0_loss) + tf.log(1 - tf.reduce_mean(prob_artist1)))
		G_loss = tf.log(1 - tf.reduce_max(prob_artist1))
		# G_loss = tf.reduce_mean(tf.log(y_label - prob_artist1))

		train_D = tf.train.AdamOptimizer(self.param.lr_d).minimize(
			D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
		train_G = tf.train.AdamOptimizer(self.param.lr_g).minimize(
			G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		flag = 0

		if self.is_plot:
			# initialize figure
			fig_num = 2
			fig_size = self.param.fig_size
			print(fig_size)
			f, a = plt.subplots(fig_num, self.param.show_img_num, figsize=(self.param.show_img_num, fig_num))
			plt.ion()   # continuously plot
			# original data (first row) for viewing
			view_data_pre = self.data.images[:self.param.show_img_num]
			# view_data_pre = np.zeros([self.param.show_img_num, 16])
			view_data = self.coder.decoder(view_data_pre)
			for i in range(self.param.show_img_num):
				a[0][i].imshow(np.reshape(view_data[i], self.param.fig_shape))
				a[0][i].set_xticks(())
				a[0][i].set_yticks(())
			print(np.argmax(self.data.labels[:self.param.show_img_num], axis=1))
			plt.draw()

		for step in range(self.param.train_times):
			real_data, labels = self.data.next_batch(self.param.batch_size)          # real painting from artist
			G_ideas = self.get_ideas()
			feed_dict = {G_in: G_ideas, real_art: real_data, y_label: labels}
			G_paintings, pa0, loss_l, d = sess.run([G_out, prob_artist0, D_loss, D0_loss, train_D, train_G]
			                                    , feed_dict)[0:4]
			if step > flag:
				precision = np.mean(np.argmax(labels, axis=1) == np.argmax(pa0, axis=1))
				print(step, ':', precision)
				flag = flag * 1.1 + 10

				feed_dict = {G_in: G_ideas, real_art: self.data.images[:self.param.show_img_num],
				             y_label:self.data.labels[:self.param.show_img_num]}
				G_paintings, pa0 = sess.run([G_out, prob_artist0], feed_dict)
				if self.is_plot:
					plot_data = self.coder.decoder(G_paintings)
					for i in range(self.param.show_img_num):
						a[1][i].clear()
						a[1][i].imshow(np.reshape(plot_data[i], self.param.fig_shape))
						a[1][i].set_xticks(())
						a[1][i].set_yticks(())
					plt.draw()
					print(np.argmax(pa0, axis=1))
					plt.pause(0.1)
			# train and get results
		if self.is_plot:
			plt.pause(10)


class Codes(object):
	def __init__(self, param=[[]] * 9):
		self.w1 = param[0]
		self.w2 = param[1]
		self.w3 = param[2]

		self.en_b1 = param[3]
		self.en_b2 = param[4]
		self.en_b3 = param[5]

		self.de_b1 = param[6]
		self.de_b2 = param[7]
		self.de_b3 = param[8]

	def encoder(self, x):
		en_1 = np.tanh(np.matmul(x, self.w1) + self.en_b1)
		en_2 = np.tanh(np.matmul(en_1, self.w2) + self.en_b2)
		en_3 = np.tanh(np.matmul(en_2, self.w3) + self.en_b3)
		return en_3

	def decoder(self, h):
		de_3 = np.tanh(np.matmul(h, np.transpose(self.w3)) + self.de_b3)
		de_2 = np.tanh(np.matmul(de_3, np.transpose(self.w2)) + self.de_b2)
		de_1 = sigmoid(np.matmul(de_2, np.transpose(self.w1)) + self.de_b1)
		return de_1


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
	param = Param(train_times=20000)
	g = gan(param, is_plot=True)
	g.get_data()
	g.train_Gan()
