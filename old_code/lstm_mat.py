# author='lwz'
# coding:utf-8
# !/usr/bin/env python3

# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import time


class loss_layer(object):
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2  # 误差函数  **=^
    # 一般来说，要使用某个类的方法，需要先实例化一个对象再调用方法。
    # 而使用@staticmethod或@classmethod，就可以不需要实例化，直接类名.方法名()来调用。

    @classmethod
    def loss2(self, pred, label):
        s = pred * np.log(label + 0.01) + (1 - pred) * np.log(1 - label + 0.01)  # 误差函数  **=^
        return s[0, 0]

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = pred[0] - label
        return diff
    # 损失函数的导函数


class LstmParam(object):  # 网络初始化
    def __init__(self, mem_cell_ct, x_dim, y_dim=1):
        concat_len = x_dim + mem_cell_ct
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.id = id
        self.cell_dim = mem_cell_ct
        # weight matrices
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)  # 随机矩阵
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wh = rand_arr(-0.1, 0.1, mem_cell_ct, mem_cell_ct)
        self.wy = rand_arr(-0.1, 0.1, y_dim, mem_cell_ct)
        # bias terms
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct, 1)
        self.bi = rand_arr(-0.1, 0.1, mem_cell_ct, 1)
        self.bf = rand_arr(-0.1, 0.1, mem_cell_ct, 1)
        self.bo = rand_arr(-0.1, 0.1, mem_cell_ct, 1)
        self.bh = rand_arr(-0.1, 0.1, mem_cell_ct, 1)
        self.by = rand_arr(-0.1, 0.1, y_dim, 1)
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi)
        self.wf_diff = np.zeros_like(self.wf)
        self.wo_diff = np.zeros_like(self.wo)
        self.wh_diff = np.zeros_like(self.wh)
        self.wy_diff = np.zeros_like(self.wy)

        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bf_diff = np.zeros_like(self.bf)
        self.bo_diff = np.zeros_like(self.bo)
        self.bh_diff = np.zeros_like(self.bh)
        self.by_diff = np.zeros_like(self.by)

    def apply_diff(self, lr=1):  # 梯度下降法修正误差
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.wh -= lr * self.wh_diff
        self.wy -= lr * self.wy_diff

        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        self.bh -= lr * self.bh_diff
        self.by -= lr * self.by_diff
        # reset diffs to zero
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi)
        self.wf_diff = np.zeros_like(self.wf)
        self.wo_diff = np.zeros_like(self.wo)
        self.wh_diff = np.zeros_like(self.wh)
        self.wy_diff = np.zeros_like(self.wy)

        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bf_diff = np.zeros_like(self.bf)
        self.bo_diff = np.zeros_like(self.bo)
        self.bh_diff = np.zeros_like(self.bh)
        self.by_diff = np.zeros_like(self.by)


class LstmState(object):
    def __init__(self, mem_cell_ct, x_dim, y_dim=1, t=100):  # 节点状态
        self.g = np.mat(np.zeros([mem_cell_ct, t]))
        self.i = np.mat(np.zeros([mem_cell_ct, t]))
        self.f = np.mat(np.zeros([mem_cell_ct, t]))
        self.o = np.mat(np.zeros([mem_cell_ct, t]))
        self.s = np.mat(np.zeros([mem_cell_ct, t + 1]))
        self.h = np.mat(np.zeros([mem_cell_ct, t + 1]))
        self.hh = np.mat(np.zeros([mem_cell_ct, t]))
        self.y = np.mat(np.zeros([y_dim, t]))  # y_len
        self.xc = np.mat(np.zeros([mem_cell_ct + x_dim, t]))  # y_len
        self.diff_h = np.mat(np.zeros([mem_cell_ct, 1]))
        self.diff_s = np.mat(np.zeros([mem_cell_ct, 1]))
        # self.diff_x = np.zeros(x_dim)

    def reset(self):
        pass


class LstmNetwork(object):
    def __init__(self, y_list, x_list, cell_dim, y_dim=1):
        self.x_list = np.mat(x_list)
        self.y_list = np.mat(y_list)
        x_dim = self.x_list.shape[0]
        y_dim = self.y_list.shape[0]
        self.x_dim = x_dim
        self.cell_dim = cell_dim
        self.y_dim = y_dim
        self.param = LstmParam(cell_dim, x_dim, y_dim)
        self.state = LstmState(cell_dim, x_dim, y_dim, self.y_list.shape[1])
        # input sequence
        self.loop_id = 0

        self.y_predict = list()
        self.loop_num = self.y_list.shape[1]
        self.lr = 0.40
        self.attenuation = 0.99
        self.max_loop = 400
        self.min_loss_diff = 0.000001
        self.this_cur_loss = 0
        self.start_time = 0

    def train(self, lr=False, maxloop=False, min_loos_diff=False, attenuation=False):
        assert self.y_list.shape[1] == self.x_list.shape[1]
        self.start_time = time.time()
        if not lr:
            lr = self.lr
        if not maxloop:
            maxloop = self.max_loop
        if not min_loos_diff:
            min_loos_diff = self.min_loss_diff
        if not attenuation:
            attenuation = self.attenuation
        lr_list = list()
        loss_list = list()
        loss_diff = 1
        cur_iter = 0
        y_pridect_list = [list() for _ in range(self.loop_num)]

        while cur_iter < maxloop and loss_diff > min_loos_diff:  # 迭代次数
            self.forward_loop()
            self.backprop_loop()  # 计算误差并且计算梯度
            loss_list.append(self.this_cur_loss[0, 0])
            if cur_iter > 1:
                loss_diff = loss_list[cur_iter - 1] - loss_list[cur_iter]
            # lr = lr0 / (1 + np.exp(cur_iter*0.0085))
            lr *= attenuation
            lr_list.append(lr)
            self.param.apply_diff(lr=lr)  # 梯度下降法 修正net
            cur_iter += 1
            for i in range(self.loop_num):
                y_pridect_list[i].append(self.state.y[0, i])
            self.reset_loop()
        print("use time:", time.time() - self.start_time)
        print("lr:", lr)
        print("loss_diff: ", loss_diff)
        print("loss: ", loss_list[cur_iter - 1])

        return loss_list, lr_list, y_pridect_list

    def feed_forward(self):  # 前向计算 前馈
        # if this is the first lstm node in the network
        # save data for use in backprop
        t = self.loop_id

        # concatenate x(t) and h(t-1)
        xc = np.vstack((self.x_list[:, t],  self.state.h[:, t]))
        self.state.g[:, t] = np.tanh(self.param.wg * xc + self.param.bg)
        self.state.i[:, t] = sigmoid(self.param.wi * xc + self.param.bi)  # input gate
        self.state.f[:, t] = sigmoid(self.param.wf * xc + self.param.bf)  # forget gate
        self.state.o[:, t] = sigmoid(self.param.wo * xc + self.param.bo)  # output gate
        self.state.s[:, t + 1] = np.multiply(self.state.g[:, t], self.state.i[:, t]) + \
                                 np.multiply(self.state.s[:, t], self.state.f[:, t])
        # s_t = new_message * input_gate + s_t-1 * forget_gate
        self.state.hh[:, t] = np.tanh(self.param.wh * self.state.s[:, t + 1] + self.param.bh)
        self.state.h[:, t + 1] = np.multiply(self.state.hh[:, t], self.state.o[:, t])
        self.state.y[:, t] = sigmoid(self.param.wy * self.state.h[:, t + 1] + self.param.by)
        self.state.xc[:, t] = xc

    def back_propagation(self):  # 后向计算
        # notice that top_diff_s is carried along the constant error carousel
        # 注意 top_diff_表示是前N个神经元通过state传递下来的常数误差
        # self.state.o * top_diff_h 表示这个神经元通过链式求导传递进来的误差
        t = self.loop_id
        loss_y = loss_layer.bottom_diff(self.state.y[:, t], self.y_list[:, t])

        dh = self.param.wy.T * loss_y + self.state.diff_h
        dhh = np.multiply(dh, self.state.o[:, t])
        dtanh_hh = 1 - np.power(self.state.hh[:, t], 2)
        ds = self.param.wh.T * np.multiply(dtanh_hh, dhh) + self.state.diff_s
        do = np.multiply(dh, self.state.hh[:, t])
        di = np.multiply(ds, self.state.g[:, t])
        dg = np.multiply(ds, self.state.i[:, t])
        df = np.multiply(ds, self.state.s[:, t])

        # diffs w.r.t. vector inside sigma / tanh function
        # diff(1/(1 + exp(-x))) = y - y*y
        di_input = np.multiply(self.state.i[:, t] - np.power(self.state.i[:, t], 2), di)
        df_input = np.multiply(self.state.f[:, t] - np.power(self.state.f[:, t], 2), df)
        do_input = np.multiply(self.state.o[:, t] - np.power(self.state.o[:, t], 2), do)
        # diff(tanh) = 1 - tanh^2 = 1 - y^2
        dg_input = np.multiply(1. - np.power(self.state.g[:, t], 2), dg)
        dh_input = np.multiply(1. - np.power(self.state.hh[:, t], 2), dhh)
        dy_input = loss_y

        # diffs w.r.t. inputs
        self.param.wi_diff += np.outer(di_input, self.state.xc[:, t])
        self.param.wf_diff += np.outer(df_input, self.state.xc[:, t])
        self.param.wo_diff += np.outer(do_input, self.state.xc[:, t])
        self.param.wg_diff += np.outer(dg_input, self.state.xc[:, t])
        self.param.wh_diff += np.outer(dh_input, self.state.s[:, t + 1])
        self.param.wy_diff += np.outer(dy_input, self.state.h[:, t + 1])

        self.param.bi_diff += di_input
        self.param.bf_diff += df_input
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input
        self.param.bh_diff += dh_input
        self.param.by_diff += dy_input

        # compute bottom diff
        dxc = np.zeros_like(self.state.xc[:, 0])
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # save bottom diffs
        self.state.diff_s += np.multiply(ds, self.state.f[:, t])
        # self.state.diff_x = dxc[:self.param.x_dim]
        self.state.diff_h += dxc[self.param.x_dim:]

    def predict(self):
        self.forward_loop()
        return self.y_predict

    def reset_loop(self):
        self.loop_id = 0
        self.state = LstmState(self.cell_dim, self.x_dim, self.y_dim, self.loop_num)
        self.this_cur_loss = 0

    def backprop_loop(self):
        """
        Updates diffs by setting target sequence
        with corresponding loss layer.
        Will *NOT* update parameters.  To update parameters,
        call self.lstm_param.apply_diff()
        计算误差并且计算梯度
        """
        # first node only gets diffs from label ...
        assert self.loop_id == self.loop_num - 1
        loss = 0
        # here s is not affecting loss due to h(t+1), hence we set equal to zero
        # self.state.diff_h = np.zeros_like(self.param.bh)
        # self.state.diff_s = np.zeros_like(self.param.bf)

        # ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        # we also propagate error along constant error carousel using diff_s
        # 按时间从后向前迭代
        while self.loop_id >= 0:
            loss += loss_layer.loss(self.state.y[:, self.loop_id], self.y_list[:, self.loop_id])
            self.back_propagation()  # 修正net
            self.loop_id -= 1
        self.loop_id = 0
        self.this_cur_loss += loss

    def forward_loop(self, x=False):
        # get index of most recent x input
        # no recurrent inputs yet
        assert self.loop_id == 0
        while self.loop_id < self.loop_num:
            self.feed_forward()
            self.loop_id += 1
        self.loop_id -= 1
        # self.lstm_node_list[self.loop_id].feed_forward(x, s_prev, h_prev)


def sigmoid(x):  # 激活函数
    return 1. / (1 + np.exp(-x))


# createst uniform random array w/ values in [a,b) and shape args
# 创建均匀分布[a, b]
# 当函数的参数不确定时，可以使用*args和**kwargs。*args没有key值，**kwargs有key值
def rand_arr(a, b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a


def example():
    x_dim = 50
    y_list = np.array([[0.7, 0.2, 0.1, 0.5, 0.8]])
    np.random.seed(0)
    x_list = np.random.random([x_dim, y_list.shape[1]])
    lstm_net = LstmNetwork(x_list=x_list, y_list=y_list, cell_dim=7)
    loss, lr, y_predict_list = lstm_net.train(lr=0.41, min_loos_diff=-999,
                                              attenuation=0.998, maxloop=400)
    plt.plot(loss)
    plt.plot(lr)
    legend = ['loss', 'lr']
    i = 0
    for y in y_predict_list:
        plt.plot(y, '.')
        i += 1
        legend.append('y_' + str(i))
    plt.legend(legend)
    plt.show()


if __name__ == "__main__":
    example()
