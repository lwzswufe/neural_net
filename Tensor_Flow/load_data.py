# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
import pyodbc
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from Tensor_Flow import MyDataSet
import pickle
import time
import random


try:
	with open('\\\\SYWG-DATA\\Share\\address.txt', 'rb') as f:
		print('connect success')
except Exception:
	pc = 'D:\\'
else:
	pc = '\\\\SYWG-DATA\\'
print('data dir:', pc)


def from_mat(code):
	fn = pc + 'data\\day0\\' + code + '.mat'
	try:
		data = sio.loadmat(fn)
	except FileNotFoundError:
		print(fn + ' is not exist')
		return False
	else:
		return data['ws']
'''
return a (X * 8) mat
'''


def from_sql(code, conn):
	a = 'SELECT * FROM ' + code
	try:
		df = pd.read_sql(sql=a, con=conn)
		# df.columns = ['date', 'status', 'open', 'high', 'low', 'close', 'vol', 'amt']
		data = np.array(df)
	except pd.io.sql.DatabaseError:
		print('SQL_table ' + code + ' is not exist')
		return False
	else:
		return data


def get_stock_list(fname='D:\\data\\S.csv'):
	df = pd.read_csv(fname, encoding='gbk')
	return df


def get_data(code, conn=None):
	if isinstance(conn, bool) or conn is None:
		return from_mat(code)
	else:
		return from_sql(code, conn)


def get_conn():
	database = 'SYWG_Day'
	server = 'SYWG-DATA\SQLEXPRESS'
	userid = 'sa'
	password = '123sa'
	driver = 'SQL Server'
	sql_sentence = 'DRIVER={' + driver + '};SERVER=' + server + ';DATABASE=' + database \
	               + ';UID=' + userid + ';PWD=' + password
	conn = pyodbc.connect(sql_sentence)
	return conn


'''
用sample_generator 制造样本
'''


def sample_generator_exm(data):
	sample = data[:, 5]
	return sample


def collect_data(sp_generator=sample_generator_exm, is_sql_con=False, limit=None,
                 end_date=None):
	df = get_stock_list()
	samples = list()
	labels = list()
	codes = list()
	dates = list()
	profits = list()
	if is_sql_con:
		conn = get_conn()
	else:
		conn = False

	if limit is None or limit == 0:
		code_list = df.code
	else:
		code_list = df.code[:limit]

	for i, code in enumerate(code_list):
		data = get_data(code, conn)
		sample, label, date_s, profit_s = sp_generator(data)
		if sample:
			samples = samples + sample
			labels = labels + label
			codes = codes + [code] * len(label)
			dates = dates + date_s
			profits = profits + profit_s
		if i % 100 == 0:
			print('get {} stocks'.format(i))

	samples = np.array(samples)
	labels = np.array(labels)

	if len(samples.shape) == 3:
		samples.reshape(samples.shape[0:2])

	if len(labels.shape) == 3:
		labels.reshape(labels.shape[0:2])

	if len(dates) > 0 and isinstance(dates[0], float):
		dates = [matlabTime_to_Python(date) for date in dates]

	c = set_datasets(samples, labels, np.array(codes), np.array(dates), np.array(profits))
	# list 类不支持list作为指针
	return c


def set_datasets(samples, labels, codes, dates, profits):
	length = samples.shape[0]
	k = list(range(length))
	random.shuffle(k)
	idx_1 = int(length * 0.5)
	idx_2 = int(length * 0.75)

	train = MyDataSet.DataSet(samples[k[:idx_1], :], labels[k[:idx_1], :],
	                          code=codes[k[:idx_1]], date=dates[k[:idx_1]],
	                          profit=profits[k[:idx_1]])
	validation = MyDataSet.DataSet(samples[k[idx_1:idx_2], :], labels[k[idx_1:idx_2], :],
	                               code=codes[k[idx_1:idx_2]], date=dates[k[idx_1:idx_2]],
	                               profit=profits[k[idx_1:idx_2]])
	test = MyDataSet.DataSet(samples[k[idx_2:], :], labels[k[idx_2:], :],
	                         code=codes[k[idx_2:]], date=dates[k[idx_2:]],
	                         profit=profits[k[idx_2:]])

	return MyDataSet.Datasets(train=train, validation=validation, test=test)


def save(data, fn=None):
	if fn is None:
		print('failed to save pickle, please input filename')
		return
	with open(fn, 'wb') as f:
		pickle.dump(data, f)


def matlabTime_to_Python(matlabTime=736864.8930):
	pythonTime = ((matlabTime - 719529) * 24 - 8) * 3600
	return time.strftime('%y-%m-%d %H:%M:%S', time.localtime(pythonTime))


def pythonTime_to_MATLAB(pythonTime=1497856238):
	if isinstance(pythonTime, str):
		if len(pythonTime) > 8:
			pythonTime = time.mktime(time.strptime(pythonTime, '%y-%m-%d %H:%M:%S'))
		else:
			pythonTime = time.mktime(time.strptime(pythonTime, '%y-%m-%d'))
	matlabTime = 719529 + (pythonTime/3600 + 8) / 24
	return matlabTime


def plot(codes, dates, length=100):
	assert len(codes) == len(dates)
	for i, code in enumerate(codes):
		data = get_data(code)
		data = data[data[:, 6] > 0, :]
		end_date = pythonTime_to_MATLAB(dates[i])
		data = data[data[:, 0] < end_date, :]
		plt.plot(np.log(data[-length:, 5]))
	plt.show()
	plt.legend(codes)


def load_matlab_dataset(fn='D:\\Cache\\macd_model.mat'):
	data = sio.loadmat(fn)
	images = data['Signal']
	labels = data['Labels']
	codes = data['Code']
	dates = data['Date']
	profits = data['Profit']

	if labels.shape[1] == 1 and sum(labels == 1) != sum(labels == 0):
		print('sample numbers in class 0 and class 1 is not equal, we wil adjust the number of classes')
		num_1 = sum(labels == 1)
		num_0 = sum(labels == 0)
		rand = np.random.random(labels.shape)
		if num_1 > num_0:
			rand[labels == 0] = 1
		else:
			rand[labels == 1] = 1
		p = abs(num_0 - num_1) / max(num_0, num_1)  # 接受概率

		labels = labels[rand > p]
		labels = np.transpose([labels, 1 - labels])
		images = images[rand.flatten() > p]

		if dates.shape[0] > 0:
			dates = dates[rand > p]
		if codes.shape[0] > 0:
			codes = codes[rand > p]
		if profits.shape[0] > 0:
			profits = profits[rand > p]

	return set_datasets(images, labels, codes, dates, profits)


if __name__ == '__main__':
	# data = from_mat('SH600000')
	print(matlabTime_to_Python())
	print(pythonTime_to_MATLAB())
	plot(['SZ002171'], ['17-03-01'])
	pass
