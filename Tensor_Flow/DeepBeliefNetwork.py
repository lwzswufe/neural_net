# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
"""
"""
from Tensor_Flow import RestrictedBoltzmannMachine as RBM
import matplotlib.pyplot as plt
import numpy as np



class DBN(object):
	def __init__(self, h_dim=[784, 196]):
		self.h_dim = np.array(h_dim)
		self.rbm = list()
		self.h_num = len(self.h_dim) - 1

	def train(self):
		param = list()
		v = None

		for i, dim in enumerate(self.h_dim[1:]):
			param.append(RBM.Param(train_times=100000, learning_rate=0.002, h_dim=dim))
			self.rbm.append(RBM.rbm(param[i], is_plot=False))
			self.rbm[i].input_data(data=v)
			self.rbm[i].train()
			v = self.rbm[i].get_next_layer_dataset()

	def get_visible(self, h):
		for i in range(len(self.h_dim[1:]) - 1, -1, -1):
			v, _ = self.rbm[i].get_visible(data=h)
			h = v
		return h

	def get_hidden(self, v=None):
		for i in range(len(self.h_dim[1:])):
			h, _ = self.rbm[i].get_hidden(data=v)
			v = h
		return v

	def show_sample(self, show_img_num=5):
		plt_num = self.h_num * 2 + 1
		f, a = plt.subplots(plt_num, show_img_num, figsize=(show_img_num, plt_num))
		width = np.round(np.sqrt(self.h_dim))
		length = np.round(np.sqrt(self.h_dim))

		plt.ion()   # continuously plot
		# original data (first row) for viewing
		view_data = self.rbm[0].data.test.images[:show_img_num]
		i = 0
		for j in range(show_img_num):
			a[i][j].imshow(np.reshape(view_data[j], (length[i], width[i])), cmap='gray')
			a[i][j].set_xticks(())
			a[i][j].set_yticks(())

		v = view_data
		for i in range(self.h_num):
			h, v_ = self.rbm[i].get_hidden(data=v)
			v = h

			for j in range(show_img_num):
				a[i+1][j].imshow(np.reshape(v_[j], (length[i+1], width[i+1])), cmap='gray')
				a[i+1][j].set_xticks(())
				a[i+1][j].set_yticks(())

		for i in range(self.h_num):
			ii = self.h_num - i - 1
			v, v_ = self.rbm[ii].get_visible(data=h)
			h = v
			for j in range(show_img_num):
				a[i+self.h_num+1][j].imshow(np.reshape(v_[j], (length[ii], width[ii])), cmap='gray')
				a[i+self.h_num+1][j].set_xticks(())
				a[i+self.h_num+1][j].set_yticks(())

		plt.draw()
		plt.show()
		plt.pause(5)
		plt.savefig('D:\\Cache\\dbn.png')
		plt.ioff()


if __name__ == '__main__':
	dbn = DBN()
	dbn.train()
	h = dbn.get_hidden()
	v = dbn.get_visible(h)
	dbn.show_sample(10)
