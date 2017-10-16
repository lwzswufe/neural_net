# author='lwz'
# coding:utf-8
"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
try: from sklearn.manifold import TSNE; HAS_SK = True
# Manifold learning方法可以认为是对诸如PCA等线性降维方法的一种扩展,以便能够处理数据中的非线性结构信息。
except: HAS_SK = False; print('\nPlease install sklearn for layer visualization\n')


class Param(object):
    def __init__(self, dataset_path='.\\MNIST_data',
                 batch_size=256, learning_rate=0.002, show_img_num=5, train_times=1000,
                 conv_1=[16, 5, 1], conv_2=[32, 5, 1], pool_1=[2, 2], pool_2=[2, 2]):
        self.batch_size = batch_size
        self.lr = learning_rate         # learning rate
        self.show_img_num = show_img_num     # 可视化显示图片数量
        self.train_times = train_times
        self.conv_1 = conv_1
        self.conv_2 = conv_2
        self.pool_1 = pool_1
        self.pool_2 = pool_2
        self.x_dim = 0
        self.dataset_path = dataset_path
        self.length = 0
        self.width = 0


def plot_with_labels(lowDWeights, labels):
    plt.cla(); X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)


class CNN(object):
    def __init__(self, param=Param(), is_plot=False, data_type='mnist'):
        self.param = param
        self.sess = ''
        self.is_plot = is_plot
        self.data_type = data_type

    def input_data(self, data=None):
        # Mnist digits
        # mnist = input_data.read_data_sets('./mnist', one_hot=False)     # use not one-hotted target data
        if data is None:
            self.data = input_data.read_data_sets(self.param.dataset_path, one_hot=True)
        else:
            self.data = data

        # plot one example
        print('images shape: ', self.data.train.images.shape)     # (55000, 28 * 28)
        print('lebal  shape: ', self.data.train.labels.shape)     # (55000, 10)
        self.param.x_dim = self.data.train.images.shape[1]
        self.param.width = int(math.sqrt(self.param.x_dim))
        self.param.length = int(math.sqrt(self.param.x_dim))
        if self.is_plot:
            plt.imshow(self.data.train.images[0].reshape((self.param.length, self.param.width)), cmap='gray')
            plt.title('i'.format(np.argmax(self.data.train.labels[0])))
            plt.show()

    def train(self):
        tf.set_random_seed(1)
        np.random.seed(1)
        tf_x = tf.placeholder(tf.float32, [None, self.param.length * self.param.width]) / 255.
        image = tf.reshape(tf_x, [-1, self.param.length, self.param.width, 1])  # (batch, height, width, channel)
        #  -1 can also be used to infer推断 the shape
        tf_y = tf.placeholder(tf.int32, [None, 10])  # input y
        # tf.nn.conv2d，一般在下载预训练好的模型时使用。

        conv1 = tf.layers.conv2d(inputs=image,  filters=self.param.conv_1[0],  kernel_size=self.param.conv_1[1],
            strides=self.param.conv_1[2], padding='same',  activation=tf.nn.relu)
        # shape (28, 28, 1)  第一组卷积层
        # inputs指需要做卷积的输入图像，它要求是一个Tensor
        # filters卷积核的数量
        # kernel_size: convolution window 卷积窗口 5*5
        # strides卷积时在图像每一维的步长，这是一个一维的向量，长度1
        # padding只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
        # 当其为‘SAME’时，表示卷积核可以停留在图像边缘
        # -> (28, 28, 16)
        # activation 正则化项

        pool1 = tf.layers.max_pooling2d(conv1, pool_size=self.param.pool_1[0], strides=self.param.pool_1[1])
        # 第一组池化层
        # the size of the pooling window 池化层大小2*2
        # 卷积时在图像每一维的步长，这是一个一维的向量，长度2
        # -> (14, 14, 16)

        conv2 = tf.layers.conv2d(pool1, self.param.conv_2[0], self.param.conv_2[1], self.param.conv_2[2]
                                 , 'same', activation=tf.nn.relu)
        # -> (14, 14, 32)
        pool2 = tf.layers.max_pooling2d(conv2, self.param.pool_2[0], self.param.pool_2[1])
        # -> (7, 7, 32)
        flat = tf.reshape(pool2, [-1, 7 * 7 * self.param.conv_2[0] ])  # -> (7*7*32, )
        output = tf.layers.dense(flat, 10)  # output layer

        loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)  # compute cost
        train_op = tf.train.AdamOptimizer(self.param.lr).minimize(loss)

        accuracy = tf.metrics.accuracy(  # return (acc, update_op), and create 2 local variables
            labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1), )[1]
        # tf.metrics.accuracy计算精度,返回accuracy和update_operation

        sess = tf.Session()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())  # the local var is for accuracy_op

        sess.run(init_op)  # initialize var in graph

        plt.ion()
        print('data max:', np.max(self.data.test.images[0]))
        test_x = self.data.test.images[:200]
        test_y = self.data.test.labels[:200]

        for step in range(self.param.train_times):
            b_x, b_y = self.data.train.next_batch(self.param.batch_size)
            _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
            if step % 100 == 0:
                # flat：第二池化层池化后数据
                # accuracy_精度
                accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
                print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

                if HAS_SK:
                    # Visualization of trained flatten layer (T-SNE)
                    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                    plot_only = 100
                    low_dim_embs = tsne.fit_transform(flat_representation[:plot_only, :])
                    labels = np.argmax(test_y, axis=1)[:plot_only]
                    plot_with_labels(low_dim_embs, labels)
        plt.ioff()

        # print 10 predictions from test data
        test_output = sess.run(output, {tf_x: test_x[:10]})
        pred_y = np.argmax(test_output, 1)
        print(pred_y, 'prediction number')
        print(np.argmax(test_y[:10], 1), 'real number')


if __name__ == '__main__':
    param = Param(train_times=1000, learning_rate=0.002, show_img_num=9)
    cnn = CNN(param)
    cnn.input_data()
    cnn.train()
    