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
                 batch_size=200, lr_g=0.002, lr_d=0.002, lr_c=0.002, show_img_num=9,
                 train_times=1000, fig_size=784, n_ideas=2, h_dim=10):
        self.batch_size = batch_size
        self.lr_d = lr_d         # learning rate
        self.lr_g = lr_g
        self.lr_c = lr_c
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

    def get_data(self, is_mnist=True):
        if is_mnist:
            with open("D:\\Cache\\ckpt_dir\\encodes_mnist.pic", 'rb') as f:
                data = pickle.load(f)
                self.data = data[0]
                self.coder = Codes(data[1])
                self.param.x_dim = self.data.images.shape[1]
                self.param.h_dim = self.data.labels.shape[1]
        else:
            with open("D:\\Cache\\ckpt_dir\\encodes.pic", 'rb') as f:
                data = pickle.load(f)
                encodes, labels, codes, dates, profits = data
                self.data = MyDataSet.DataSet(images=encodes, labels=labels, code=codes, date=dates, profit=profits)
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

        real_art = tf.placeholder(tf.float32, [None, self.param.x_dim], name='real_in')
        # receive art work from the famous artist
        with tf.variable_scope('Discriminator'):
            D_l0 = tf.layers.dense(real_art, self.param.x_dim, tf.nn.relu, name='l')
            prob_real = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name='out')              # probability that the art work is made by artist
            # reuse layers for generator
            D_l1 = tf.layers.dense(G_out, self.param.x_dim, tf.nn.relu, name='l', reuse=True)            # receive art work from a newbie like G
            prob_fake = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out', reuse=True)  # probability that the art work is made by artist

        with tf.variable_scope('Classifier'):
            C_l0 = tf.layers.dense(real_art, self.param.x_dim, tf.nn.relu, name='c_in')
            prob_real_class = tf.layers.dense(C_l0, self.param.h_dim, tf.nn.sigmoid, name='c_out')

            C_l1 = tf.layers.dense(G_out, self.param.x_dim, tf.nn.relu, name='c_in', reuse=True)            # receive art work from a newbie like G
            prob_fake_class = tf.layers.dense(C_l1, self.param.h_dim, tf.nn.sigmoid, name='c_out', reuse=True)  # probability that the art work is made by artist

        y_label = tf.placeholder(tf.float32, [None, self.param.h_dim])
        z = tf.placeholder(tf.float32, [None, 1])

        # D0_loss = tf.diag_part(tf.matmul(prob_artist0, y_label, transpose_b=True))
        real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(z), logits=prob_real)
        fake_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(z), logits=prob_fake)
        g_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(z), logits=prob_fake)
        c_real_loss = tf.losses.softmax_cross_entropy(onehot_labels=y_label, logits=prob_real_class)
        c_fake_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.ones_like(y_label), logits=prob_fake_class)
        #c_fake_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.reduce_max(tf.ones_like(y_label), axis=1),
        #                                              logits=tf.reduce_max(prob_fake_class, axis=1))

        D_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)
        G_loss = tf.reduce_mean(g_loss) + tf.reduce_mean(c_fake_loss)
        C_loss = tf.reduce_mean(c_real_loss)

        # G_loss = tf.reduce_mean(tf.log(y_label - prob_artist1))

        train_D = tf.train.AdamOptimizer(self.param.lr_d).minimize(
            D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
        train_G = tf.train.AdamOptimizer(self.param.lr_g).minimize(
            G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))
        train_C = tf.train.AdamOptimizer(self.param.lr_c).minimize(
            C_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Classifier'))

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
            feed_dict = {G_in: G_ideas, real_art: real_data, y_label: labels, z: np.ones([labels.shape[0], 1])}
            G_paintings, pa0, p_f = sess.run([G_out, prob_real_class, prob_fake,
                                             train_D, train_G, train_C], feed_dict)[0:3]
            if step > flag:
                precision = np.mean(np.argmax(labels, axis=1) == np.argmax(pa0, axis=1))
                print(step, ':', precision, '  ', np.mean(p_f))
                flag = flag * 1.1 + 10

                feed_dict = {G_in: G_ideas, real_art: self.data.images[:self.param.show_img_num],
                             y_label:self.data.labels[:self.param.show_img_num], z: np.ones([labels.shape[0], 1])}
                G_paintings, pa0 = sess.run([G_out, prob_real_class], feed_dict)
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

        feed_dict = {real_art: self.data.images, y_label: self.data.labels, z: np.ones([self.data.labels.shape[0], 1])}
        pa0 = sess.run(prob_real_class, feed_dict)
        precision = np.mean(np.argmax(self.data.labels, axis=1) == np.argmax(pa0, axis=1))
        print(np.mean(self.data.profit))
        print(np.mean(self.data.profit[np.argmax(pa0, axis=1) < 1]))
        print(precision)
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
    param = Param(train_times=20000, lr_d=0.001, lr_g=0.001, lr_c=0.001)
    g = gan(param, is_plot=False)
    g.get_data(is_mnist=False)
    g.train_Gan()
