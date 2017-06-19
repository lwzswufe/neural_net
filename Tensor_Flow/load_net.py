# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
import tensorflow as tf


model_file = 'D:\\Cache\\ckpt'
# 训练中保存的模型
tf.convert_to_tensor_or_sparse_tensor(11,11)
data_file = 'D:\\Cache\\ckpt_dirmodel.ckpt.meta'
# 训练中保存的数据
with tf.Session() as sess:
	# tf.initialize_all_variables().run()
	tf.global_variables_initializer().run()

	ckpt = tf.train.get_checkpoint_state(model_file)
	if ckpt and ckpt.model_checkpoint_path:
		print(ckpt.model_checkpoint_path)
		tf.train.Saver.restore(sess, ckpt.model_checkpoint_path)
		print('load over')
	else:
		print('load fail')
