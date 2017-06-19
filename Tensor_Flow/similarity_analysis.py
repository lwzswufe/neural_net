# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
import pickle
import numpy as np
from Tensor_Flow import load_data


class similarity(object):
	def __init__(self, fn='D:\\Cache\\encodes.pic'):
		with open(fn, 'rb') as f:
			data = pickle.load(f)
		self.encodes = data[0]
		self.code = data[1]
		self.date = data[2]

	def distance(self, i, j):
		lx = np.sqrt(self.encodes[i, :] @ self.encodes[i, :])
		ly = np.sqrt(self.encodes[j, :] @ self.encodes[j, :])
		return 1 - self.encodes[i, :] @ self.encodes[j, :] / (lx * ly)

	def get_distances_from(self, i):
		d = [self.distance(i, j) for j in range(len(self.code))]
		return np.array(d)

	def sorted_distance(self, i):
		d = self.get_distances_from(i)
		rank = d.argsort()
		return d, rank

	def get_nearest(self, i=None, num=5, d=None, rank=None):
		if d is None and i is None:
			print('input error')
		if d is None:
			d, rank = self.sorted_distance(i)

		if i is not None:
			print("target: code: {} date: {}".format(self.code[i], self.date[i]))

		for j in range(num):
			idx = round(rank[j])
			print("rank:{} code: {} date: {} distance:{:.4f}".format(j, self.code[idx], self.date[idx], d[idx]))

	def plot_items(self, ids):
		code_list = list()
		date_list = list()
		for id in ids:
			code_list.append(self.code[id])
			date_list.append(self.date[id])
		load_data.plot(code_list, date_list)

	def plot_nearest(self, i, num=7):
		d, rank = self.sorted_distance(i)
		self.get_nearest(i, num, d, rank)
		self.plot_items(rank[:num])

	def find(self, code, date):
		for i, code_ in enumerate(self.code):
			if code == code_ and self.date[i] == date:
				return i
		print('do not find code {} date {}'.format(code, date))


if __name__ == '__main__':
	s = similarity()
	d, rank = s.sorted_distance(203)
	s.get_nearest(i=203, d=d, rank=rank)
	pass