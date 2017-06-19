# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
import numpy
import pickle
import os
import collections
# from tensorflow.contrib.learn.python.learn import datasets


# Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
class Datasets(object):
	def __init__(self, train, validation, test):
		self.train = train
		self.validation = validation
		self.test = test

	def get_all(self):
		all_images = numpy.concatenate((self.train.images, self.validation.images, self.test.images))
		all_labels = numpy.concatenate((self.train.labels, self.validation.labels, self.test.labels))
		all_codes = numpy.concatenate((self.train.code, self.validation.code, self.test.code))
		all_dates = numpy.concatenate((self.train.date, self.validation.date, self.test.date))
		return all_images, all_labels, all_codes, all_dates


class DataSet(object):

	def __init__(self, images, labels, fake_data=False, one_hot=False, code=[], date=[]):
		"""Construct a DataSet. one_hot arg is used only if fake_data is true."""

		if fake_data:
			self._num_examples = 10000
			self.one_hot = one_hot
		else:
			assert images.shape[0] == labels.shape[0], (
				'images.shape: %s labels.shape: %s' % (images.shape,labels.shape))
			self._num_examples = images.shape[0]
			if len(images.shape) >= 3:
				self.image_size = images.shape[1] * images.shape[2]
				assert images.shape[3] == 1
				images = images.reshape(images.shape[0], self.image_size)
			else:
				self.image_size = images.shape[1]
			self.lebal_size = labels.shape[1]
			# Convert from [0, 255] -> [0.0, 1.0].
			images = images.astype(numpy.float32)
			if len(code) == 0:  # 导入图片数据
				images = numpy.multiply(images, 1.0 / 255.0)

		if len(code) == 0 and not fake_data:
			self.code = None
		else:
			assert len(code) == self._num_examples
			self.code = code

		if len(date) == 0 and not fake_data:
			self.date = None
		else:
			assert len(date) == self._num_examples
			self.date = date

		self._images = images
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size, fake_data=False):
		"""Return the next `batch_size` examples from this data set."""
		if fake_data:
			fake_image = [1] * self.image_size
			if self.one_hot:
				fake_label = [1] + [0] * (self.lebal_size - 1)
			else:
				fake_label = 0
			return [fake_image for _ in range(batch_size)], [
				fake_label for _ in range(batch_size)]
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = numpy.arange(self._num_examples)
			numpy.random.shuffle(perm)
			self._images = self._images[perm]
			self._labels = self._labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._images[start:end], self._labels[start:end]

	def __len__(self):
		return self._num_examples


def load_pick(fn=None):
	if fn is None or not os.path.exists(fn):
		print('file：', fn, ' is not exist')
		return None
	with open(fn, 'rb') as f:
		return pickle.load(f)


if __name__ == '__main__':
	c = DataSet(numpy.zeros([100, 28, 28, 1]), numpy.ones([100, 10]))
	sample, lebal = c.next_batch(5)
	print(sample)
	print(lebal)
	c = DataSet(numpy.zeros([100, 784]), numpy.ones([100, 1]))
	sample, lebal = c.next_batch(5)
	print(sample)
	print(lebal)
