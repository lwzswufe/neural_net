# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
import pickle
import numpy as np
from Tensor_Flow import load_data
import pandas as pd
import time


class similarity(object):
	def __init__(self, fn='D:\\Cache\\encodes.pic'):
		with open(fn, 'rb') as f:
			data = pickle.load(f)
		self.encodes = data[0]
		self.code = data[1]
		self.date = data[2]
		self.profits = data[3]
		self.class_encodes = list()
		self.class_num = []
		self.class_list = []

	def distance(self, i, j, i_isclass=False, j_isclass=False):
		if i_isclass:
			x = self.class_encodes[i]
		else:
			x = self.encodes[i, :]

		if j_isclass:
			y = self.class_encodes[j]
		else:
			y = self.encodes[j, :]

		distance = 1 - x @ y / (np.sqrt(x @ x) * np.sqrt(y @ y))
		return distance

	def get_distances_from(self, i, max_distance=None, class_id=0):
		if max_distance is None:
			d = [self.distance(i, j) for j in range(len(self.code))]
			return np.array(d)
		else:
			distance = list()
			idxs = list()
			for j in self.class_list[class_id]:
				d = self.distance(i, j)
				if d < max_distance:
					distance.append(d)
					idxs.append(j)
			return np.array(distance), np.array(idxs)

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
			print("rank:{} code: {} date: {} distance:{:.4f} profit:{:.4f}".format(
				j, self.code[idx], self.date[idx], d[idx], self.profits[idx]))

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
		return None

	def classify(self, class_distance=0.3):
		st_time = time.time()
		class_list = [[0]]
		self.class_encodes.append(self.encodes[0, :])

		for i, code in enumerate(self.code[1:]):
			for j, class_ in enumerate(class_list):
				distance = self.distance(i, j, j_isclass=True)
				if distance < class_distance:
					class_list[j].append(i)
					break
			else:
				class_list.append([i])
				self.class_encodes.append(self.encodes[i, :])
			if i % 10000 == 0:
				print('{}times'.format(i))

		self.class_num = [len(class_) for class_ in class_list]
		self.class_list = [set(class_) for class_ in class_list]
		print("classify used {:.4f} secounds".format(time.time() - st_time))

	def find_class(self, i):
		for j, class_ in enumerate(self.class_list):
			if i in class_:
				return j
		return False

	def recommend(self, date):
		profit_mean = list()
		profit_std = list()
		distance_mean = list()
		win_ratio = list()

		code_list = get_stock_list()
		flag = 1
		date_list = np.array(range(len(self.date)))[self.date == date]
		for i in date_list:
			code = self.code[i]
			j = self.find_class(i)
			if j:
				p_mean, p_std, d_mean, w_ratio = self.expected_profit(i, num=40, date=date, class_id=j)
				if p_mean is not None:
					profit_mean.append(p_mean)
					profit_std.append(p_std)
					distance_mean.append(d_mean)
					win_ratio.append(w_ratio)
				else:
					print('do not find near {}'.format(code))
			flag += 1
			if flag % 10 == 0:
				print(flag)

		profit_mean_arr = -np.array(profit_mean)
		rank = np.argsort(profit_mean_arr)

		for i in range(40):
			idx = rank[i]
			print("rank:{} code: {} mean_distance:{:.4f} mean_profit:{:.4f} profit std:{:.4f}"
			      " true profit:{:.4f} win ratio:{:.4f}".format(i, code_list[idx], distance_mean[idx],
			      profit_mean[idx], profit_std[idx], self.profits[date_list[idx]], win_ratio[idx]))

	def expected_profit(self, i, num=40, date='00-01-01 00:00:00', class_id=0):
		d, idxs = self.get_distances_from(i, max_distance=0.03, class_id=class_id)
		if len(d) == 0:
			return None, None, None, 0

		rank = d.argsort()
		'''
		rank: index in d
		idxs[idx]: index in self.date self.code
		'''

		profit = list()
		weight = list()
		distance = list()
		flag = 0
		while flag < min(num, len(rank)):
			idx = round(rank[flag])
			if date == self.date[idxs[idx]]:
				flag += 1
				continue
			else:
				flag += 1
			profit.append(self.profits[idxs[idx]])
			weight.append(get_weight(d[idx]))
			distance.append(d[idx])

		if len(profit) <= 5:
			return None, None, None, 0

		weight /= sum(weight)
		profit = np.array(profit)
		profit_mean = profit @ np.array(weight)
		profit_std = np.std(profit)
		distance_mean = np.mean(np.array(distance))
		win_ratio = sum(profit > 0.0) / len(profit)
		return profit_mean, profit_std, distance_mean, win_ratio


def get_weight(d):
	weight = 0.01 / (0.01 + d)
	return weight


def get_stock_list(fname='D:\\data\\S.csv'):
	df = pd.read_csv(fname, encoding='gbk')
	return np.array(df.code)


if __name__ == '__main__':
	s = similarity()
	d, rank = s.sorted_distance(203)
	s.get_nearest(i=203, d=d, rank=rank)
	# s.plot_nearest(2, num=21)
	s.classify()
	s.recommend(date='17-06-23 00:00:00')
	pass