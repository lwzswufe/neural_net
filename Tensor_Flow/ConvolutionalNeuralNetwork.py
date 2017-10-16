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

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001              # learning rate

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)  # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# plot one example
print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape)     # (55000, 10)
plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
plt.title('i'.format(np.argmax(mnist.train.labels[0])))
plt.show()

tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.
image = tf.reshape(tf_x, [-1, 28, 28, 1])               # (batch, height, width, channel)
#  -1 can also be used to infer推断 the shape
tf_y = tf.placeholder(tf.int32, [None, 10])             # input y

# CNN
# tf.layers.conv2d参数丰富，一般用于从头训练一个模型。
# tf.nn.conv2d，一般在下载预训练好的模型时使用。
conv1 = tf.layers.conv2d(   # shape (28, 28, 1)  第一组卷积层
    inputs=image,  # 指需要做卷积的输入图像，它要求是一个Tensor
    filters=16,  # 卷积核的数量
                 # 相当于CNN中的卷积核，它要求是一个Tensor [filter_height, filter_width, in_channels, out_channels]
                 # 这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
    kernel_size=5,  # convolution window 卷积窗口 5*5
    strides=1,  # 卷积时在图像每一维的步长，这是一个一维的向量，长度1
    padding='same',  # 只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
                      # 当其为‘SAME’时，表示卷积核可以停留在图像边缘
    activation=tf.nn.relu)   # 正则化项
    # -> (28, 28, 16)

pool1 = tf.layers.max_pooling2d(  # 第一组池化层
    conv1,
    pool_size=2,     # the size of the pooling window 池化层大小2*2
    strides=2,)      # 卷积时在图像每一维的步长，这是一个一维的向量，长度2
    # -> (14, 14, 16)

conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)    # -> (14, 14, 32)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (7, 7, 32)


flat = tf.reshape(pool2, [-1, 7*7*32])          # -> (7*7*32, )
output = tf.layers.dense(flat, 10)              # output layer

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]
# tf.metrics.accuracy计算精度,返回accuracy和update_operation

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
# Manifold learning方法可以认为是对诸如PCA等线性降维方法的一种扩展,以便能够处理数据中的非线性结构信息。
except: HAS_SK = False; print('\nPlease install sklearn for layer visualization\n')


def plot_with_labels(lowDWeights, labels):
    plt.cla(); X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()
for step in range(600):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:
        # flat：第二池化层池化后数据
        # accuracy_精度
        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

        if HAS_SK:
            # Visualization of trained flatten layer (T-SNE)
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000); plot_only = 500
            low_dim_embs = tsne.fit_transform(flat_representation[:plot_only, :])
            labels = np.argmax(test_y, axis=1)[:plot_only]; plot_with_labels(low_dim_embs, labels)
plt.ioff()

# print 10 predictions from test data
test_output = sess.run(output, {tf_x: test_x[:10]})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:10], 1), 'real number')