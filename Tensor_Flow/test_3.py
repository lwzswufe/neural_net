# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
'''
restore model from file
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

sess = tf.Session()

new_saver = tf.train.import_meta_graph("D:\\Cache\\ckpt_dir\\.meta")
new_saver.restore(sess, save_path="D:\\Cache\\ckpt_dir\\")

# graph = tf.get_default_graph()
tf_x = sess.graph.get_tensor_by_name("tf_x:0")
decoded = sess.graph.get_tensor_by_name("decoded:0")
encoded = sess.graph.get_tensor_by_name("encoded:0")

with open("D:\\Cache\\ckpt_dir\\encodes_mnist.pic", 'rb') as f:
	data = pickle.load(f)

# sess.run(tf.global_variables_initializer())
decodes = sess.run(decoded, {encoded: data[0].images[5:7]})

plt.imshow(decodes[0].reshape(28, 28), cmap='gray')
plt.show()



