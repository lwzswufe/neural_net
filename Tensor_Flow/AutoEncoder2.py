# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
"""
AutoEncoder without tensorflow.layers
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
                 batch_size=256, learning_rate=0.002, show_img_num=5, train_times=1000, h_dim=[225, 49, 16]):
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
    def __init__(self, param, is_plot=False, data_type='mnist'):
        self.param = param
        self.sess = ''
        self.is_plot = is_plot
        self.data_type = data_type

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

    def train(self):
        tf.set_random_seed(1)
        # tf placeholder
        std = 0.005
        tf_x = tf.placeholder(tf.float32, [None, self.param.x_dim], name='tf_x')    # value in the range of (0, 1)

        w_1 = tf.Variable(tf.random_normal([self.param.x_dim, self.param.h_dim_1], stddev=std), tf.float32)
        w_2 = tf.Variable(tf.random_normal([self.param.h_dim_1, self.param.h_dim_2], stddev=std), tf.float32)
        w_3 = tf.Variable(tf.random_normal([self.param.h_dim_2, self.param.h_dim_3], stddev=std), tf.float32)

        b_en_1 = tf.Variable(tf.random_normal([self.param.h_dim_1], stddev=std), tf.float32)
        b_en_2 = tf.Variable(tf.random_normal([self.param.h_dim_2], stddev=std), tf.float32)
        b_en_3 = tf.Variable(tf.random_normal([self.param.h_dim_3], stddev=std), tf.float32)

        b_de_1 = tf.Variable(tf.random_normal([self.param.x_dim], stddev=std), tf.float32)
        b_de_2 = tf.Variable(tf.random_normal([self.param.h_dim_1], stddev=std), tf.float32)
        b_de_3 = tf.Variable(tf.random_normal([self.param.h_dim_2], stddev=std), tf.float32)

        # encoder
        en_1 = tf.nn.tanh(tf.matmul(tf_x, w_1) + b_en_1)
        en_2 = tf.nn.tanh(tf.matmul(en_1, w_2) + b_en_2)
        en_3 = tf.nn.tanh(tf.matmul(en_2, w_3) + b_en_3, name='encoded')

        # decoder
        de_3 = tf.nn.tanh(tf.matmul(en_3, tf.transpose(w_3)) + b_de_3)
        de_2 = tf.nn.tanh(tf.matmul(de_3, tf.transpose(w_2)) + b_de_2)
        de_1 = tf.nn.sigmoid(tf.matmul(de_2, tf.transpose(w_1)) + b_de_1, name='decoded')

        # train & loss
        loss = tf.losses.mean_squared_error(labels=tf_x, predictions=de_1)
        train = tf.train.AdamOptimizer(self.param.lr).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        self.sess = sess

        if self.is_plot and self.data_type == 'mnist':
            # initialize figure
            fig_num = 7
            fig_size = [int(np.sqrt(self.param.h_dim_1)), int(np.sqrt(self.param.h_dim_2)),
                        int(np.sqrt(self.param.h_dim_3)), int(np.sqrt(self.param.h_dim_2)),
                        int(np.sqrt(self.param.h_dim_1)), int(np.sqrt(self.param.x_dim))]
            print(fig_size)
            f, a = plt.subplots(fig_num, self.param.show_img_num, figsize=(self.param.show_img_num, fig_num))
            plt.ion()   # continuously plot
            # original data (first row) for viewing
            view_data = self.data.test.images[:self.param.show_img_num]
            for i in range(self.param.show_img_num):
                a[0][i].imshow(np.reshape(view_data[i], (self.param.length, self.param.width)), cmap='gray')
                a[0][i].set_xticks(())
                a[0][i].set_yticks(())

        for step in range(self.param.train_times):
            b_x, b_y = self.data.train.next_batch(self.param.batch_size)
            _,  decoded_, loss_, bias = sess.run([train, de_1, loss, w_1], {tf_x: b_x})

            if step % 100 == 0 and self.is_plot and self.data_type == 'mnist':     # plotting
                print('train loss: %.4f' % loss_)
                # plotting decoded image (second row)
                plot_data = sess.run([en_1, en_2, en_3, de_3, de_2, de_1], {tf_x: view_data})
                for fig_id in range(fig_num - 1):
                    for i in range(self.param.show_img_num):
                        a[fig_id+1][i].clear()
                        a[fig_id+1][i].imshow(np.reshape(plot_data[fig_id][i], (fig_size[fig_id],
                                                                                fig_size[fig_id])) )
                        a[fig_id+1][i].set_xticks(())
                        a[fig_id+1][i].set_yticks(())
                plt.draw()
                plt.pause(0.01)
            elif step % 100 == 0:
                print('%s step train loss: %.4f' % (str(step), loss_))

        if self.is_plot:
            plt.ioff()

        if self.data_type == 'mnist':
            encodes = sess.run(en_3, feed_dict={tf_x: self.data.train.images})
            train_data = MyDataSet.DataSet(images=encodes, labels=self.data.train.labels)

            view_data = self.data.test.images[:2]
            layers_param = sess.run([w_1, w_2, w_3, b_en_1, b_en_2, b_en_3, b_de_1, b_de_2, b_de_3],
                                    {tf_x: view_data})

            with open('D:\\Cache\\ckpt_dir\\encodes_mnist.pic', 'wb') as f:
                pickle.dump([train_data, layers_param], f)
            saver.save(sess, "D:\\Cache\\ckpt_dir\\")
            print('save sess over')
        elif self.data_type == 'stock':
            all_images, all_labels, all_codes, all_dates, all_profits = self.data.get_all()
            encodes = sess.run(en_3, feed_dict={tf_x: all_images})
            with open('D:\\Cache\\ckpt_dir\\encodes.pic', 'wb') as f:
                pickle.dump([encodes, all_labels, all_codes, all_dates, all_profits], f)
            print('save stock over')


def sample_generator(data, n_day=5):
    data = data[data[:, 7] > 0, :]
    data = data[:-1]
    length = data.shape[0]
    sample_length = 80
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
            Closes = data[st: ed, 5]
            Vols = data[st: ed, 6]
            Atr = data[st: ed, 3] - data[st: ed, 4]
            Closes = Closes - np.mean(Closes) + 0.5
            Atr = Atr - np.mean(Atr) + 0.5
            Vols = Vols - np.mean(Vols) + 0.5
            assert Closes.shape[0] == sample_length
            # sample.append(np.concatenate((Closes, Vols, Atr), axis=0))
            sample.append(Closes)
            label.append(np.ones([1, 1]))
            date_s.append(data[ed - 1, 0])
            nday_profit = data[min(ed - 1 + n_day, length - 1), 5] - data[ed - 1, 5]
            profit_s.append(nday_profit)
            st, ed = st + interval, ed + interval
    return sample, label, date_s, profit_s


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def daily():
    c = load_data.load_matlab_dataset()  # 股票交易模型
    # c = load_data.collect_data(sample_generator, limit=0)  # 股票相似度
    load_data.save(c, fn)
    param = Param(train_times=20000, learning_rate=0.002, show_img_num=9)
    ae = AutoEncoder(param, is_plot=False, data_type='stock')
    ae.input_data(data=c)
    ae.train()


if __name__ == '__main__':
    fn = 'D:\\Cache\\autoencoder.pic'
    if os.path.exists(fn) and False:
        c = MyDataSet.load_pick(fn)
    else:
        # c = load_data.load_matlab_dataset()  # 股票交易模型
        c = load_data.collect_data(sample_generator, limit=0)  # 股票相似度
        load_data.save(c, fn)
    param = Param(train_times=2000, learning_rate=0.002, show_img_num=9)
    ae = AutoEncoder(param, is_plot=False, data_type='stock')
    ae.input_data(data=c)
    ae.train()
