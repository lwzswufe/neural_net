# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from Tensor_Flow import MyDataSet
from tensorflow.examples.tutorials.mnist import input_data

'''
GAN识别手写数字集合
'''


class Param(object):
    '''
     参数类
    '''
    def __init__(self, dataset_path='.\\MNIST_data',
                 batch_size=200, lr_g=0.002, lr_d=0.002, lr_c=0.002, lr_a=0.002,
                 show_img_num=10, train_times=1000, fig_size=784, n_ideas=7):
        self.batch_size = batch_size
        self.lr_d = lr_d         # learning rate
        self.lr_g = lr_g
        self.lr_c = lr_c
        self.lr_a = lr_a
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
        self.fake_h_dim = self.h_dim + self.n_ideas


class gan(object):
    '''
    GAN网络
    '''
    def __init__(self, param=Param(), is_plot=False):
        self.param = param
        self.data = ''
        self.coder = Codes()  # 自动编码器
        self.is_plot = is_plot
        self.is_mnist = True
        pass

    def get_data(self):
        if self.is_mnist:    # 是否是原始mnist数据集
            mnist = input_data.read_data_sets('.\\MNIST_data', one_hot=True)
            self.data = mnist[0]              # 训练集
            self.test_data = mnist[1]         # 测试集
            self.coder = None
        else:
            with open("D:\\Cache\\ckpt_dir\\encodes_mnist.pic", 'rb') as f:
                data = pickle.load(f)         # 加载通过自动编码器"编码"后的数据
                self.data = data[0]
                self.coder = Codes(data[1])

        self.param.x_dim = self.data.images.shape[1]
        self.param.h_dim = self.data.labels.shape[1]
        print('h_dim: {}, x_dim: {}'.format(self.param.h_dim, self.param.x_dim))

    def get_ideas(self):     # painting from the famous artist (real target)
        '''
        获取idea  标签+随机数 标签为0-9固定顺序
        :return:
        '''
        random_data = 0.1 * np.random.randn(self.param.batch_size, self.param.n_ideas)
        eye_label = np.eye(self.param.h_dim, self.param.h_dim)
        fake_label = np.repeat(eye_label, int(self.param.batch_size/self.param.h_dim)-1, axis=0)
        fake_label = np.concatenate((eye_label, fake_label), axis=0)
        fake_data = np.hstack([fake_label, random_data])
        return fake_label, fake_data

    def get_ideas_same(self, label):     # painting from the famous artist (real target)
        '''
        获取idea  标签+随机数 标签同真实数据
        :return:
        '''
        random_data = 0.1 * np.random.randn(label.shape[0], self.param.n_ideas)
        fake_label = label
        fake_data = np.hstack([fake_label, random_data])
        return fake_label, fake_data

    def train_Gan(self):
        real_art = tf.placeholder(tf.float32, [None, self.param.x_dim], name='real_in')

        with tf.variable_scope('Generator'):
            # 生成器
            R_en_1 = tf.layers.dense(real_art, 200, tf.nn.tanh)
            R_en_2 = tf.layers.dense(R_en_1, 60, tf.nn.tanh)
            R_en_3 = tf.layers.dense(R_en_2, self.param.fake_h_dim, tf.nn.tanh)

            R_de_3 = tf.layers.dense(R_en_3, 60, tf.nn.tanh, name='de_3')
            R_de_2 = tf.layers.dense(R_de_3, 200, tf.nn.tanh, name='de_2')
            R_de_1 = tf.layers.dense(R_de_2, self.param.x_dim, tf.nn.sigmoid, name='de_1')

            G_label = tf.placeholder(tf.float32, [None, self.param.h_dim])          # random ideas (could from normal distribution)
            G_idea = tf.placeholder(tf.float32, [None, self.param.fake_h_dim])
            G_de_1 = tf.layers.dense(G_idea, 60, tf.nn.tanh)
            G_de_2 = tf.layers.dense(G_de_1, 200, tf.nn.tanh, name='de_2', reuse=True)
            G_out = tf.layers.dense(G_de_2, self.param.x_dim, tf.nn.sigmoid, name='de_1', reuse=True)

        # 真实数据
        with tf.variable_scope('Discriminator'):
            # 判别器
            D_l0 = tf.layers.dense(real_art, self.param.x_dim, tf.nn.relu, name='l')
            prob_real = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name='out')
            # 真实数据被判定为真的概率
            D_l1 = tf.layers.dense(G_out, self.param.x_dim, tf.nn.relu, name='l', reuse=True)
            prob_fake = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out', reuse=True)
            # 伪造数据被判定为真的概率

        with tf.variable_scope('Classifier'):
            # 分类器
            C_l0 = tf.layers.dense(real_art, self.param.x_dim, tf.nn.relu, name='c_in')
            prob_real_class = tf.layers.dense(C_l0, self.param.h_dim, tf.nn.sigmoid, name='c_out')
            # 真数据识别为0-9的概率
            C_l1 = tf.layers.dense(G_out, self.param.x_dim, tf.nn.relu, name='c_in', reuse=True)            # receive art work from a newbie like G
            prob_fake_class = tf.layers.dense(C_l1, self.param.h_dim, tf.nn.sigmoid, name='c_out', reuse=True)  # probability that the art work is made by artist
            # 伪造数据识别为0-9的概率

        y_label = tf.placeholder(tf.float32, [None, self.param.h_dim])
        # 真数据标签
        z = tf.placeholder(tf.float32, [None, 1])
        # 伪造数据鉴别结果数组

        # D0_loss = tf.diag_part(tf.matmul(prob_artist0, y_label, transpose_b=True))
        real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(z), logits=prob_real)
        fake_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(z), logits=prob_fake)
        g_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(z), logits=prob_fake)
        c_real_loss = tf.losses.softmax_cross_entropy(onehot_labels=y_label, logits=prob_real_class)
        c_fake_loss = tf.losses.softmax_cross_entropy(onehot_labels=G_label, logits=prob_fake_class)

        D_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)
        G_loss = tf.reduce_mean(g_loss) + tf.reduce_mean(c_fake_loss)
        C_loss = tf.reduce_mean(c_real_loss)
        A_loss = tf.losses.mean_squared_error(labels=real_art, predictions=R_de_1)

        # G_loss = tf.reduce_mean(tf.log(y_label - prob_artist1))

        train_D = tf.train.AdamOptimizer(self.param.lr_d).minimize(
            D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
        train_G = tf.train.AdamOptimizer(self.param.lr_g).minimize(
            G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))
        train_C = tf.train.AdamOptimizer(self.param.lr_c).minimize(
            C_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Classifier'))
        train_A = tf.train.AdamOptimizer(self.param.lr_a).minimize(
            A_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

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
            if not self.is_mnist:
                view_data = self.coder.decoder(view_data_pre)
            else:
                view_data = view_data_pre

            for i in range(self.param.show_img_num):
                a[0][i].imshow(np.reshape(view_data[i], self.param.fig_shape))
                a[0][i].set_xticks(())
                a[0][i].set_yticks(())
            print(np.argmax(self.data.labels[:self.param.show_img_num], axis=1))
            plt.draw()

        for step in range(self.param.train_times):
            real_data, labels = self.data.next_batch(self.param.batch_size)          # real painting from artist
            # G_labels, G_ideas = self.get_ideas()
            G_labels, G_ideas = self.get_ideas_same(labels)
            feed_dict = {G_label: G_labels, G_idea: G_ideas,  real_art: real_data, y_label: labels, z: np.zeros([labels.shape[0], 1])}
            G_paintings, pa0, p_f = sess.run([G_out, prob_real_class, prob_fake, train_D, train_G, train_C, train_A]
                                                , feed_dict)[:3]
            if step > flag:
                if self.is_mnist:
                    real_data, labels = self.test_data.images[:500], self.test_data.labels[:500]
                else:
                    real_data, labels = self.data.next_batch(self.param.batch_size*10)  # real painting from artist
                # G_labels, G_ideas = self.get_ideas()
                G_labels, G_ideas = self.get_ideas_same(labels)

                feed_dict = {G_label: G_labels,
                             G_idea: G_ideas,
                             real_art: real_data,
                             y_label: labels,
                             z: np.zeros([labels.shape[0], 1])}
                G_paintings, pa0, g_label = sess.run([G_out, prob_real_class, G_label], feed_dict)

                precision = np.mean(np.argmax(labels, axis=1) == np.argmax(pa0, axis=1))
                print(step, ':', precision, "fake_ratio: ", np.mean(p_f))
                flag = flag * 1.1 + 10

                if self.is_plot:
                    if self.is_mnist:
                        plot_data = G_paintings
                    else:
                        plot_data = self.coder.decoder(G_paintings)

                    for i in range(self.param.show_img_num):
                        a[1][i].clear()
                        a[1][i].imshow(np.reshape(plot_data[i], self.param.fig_shape))
                        a[1][i].set_xticks(())
                        a[1][i].set_yticks(())

                    plt.draw()
                    # print(np.argmax(self.data.labels[:self.param.show_img_num], axis=1))
                    # print(np.argmax(pa0, axis=1))
                    # print(np.argmax(g_label, axis=1))
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
    param = Param(train_times=2000, lr_c=0.001, lr_g=0.001, lr_d=0.0002, lr_a=0.002, batch_size=200)
    g = gan(param, is_plot=True)
    g.get_data()
    g.train_Gan()
