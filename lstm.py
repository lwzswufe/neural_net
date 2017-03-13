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
        if label is None:
            return 0
        else:
            return (pred[0] - label) ** 2  # 误差函数  **=^
    # 一般来说，要使用某个类的方法，需要先实例化一个对象再调用方法。
    # 而使用@staticmethod或@classmethod，就可以不需要实例化，直接类名.方法名()来调用。

    @classmethod
    def loss2(self, pred, label):
        s = pred * np.log(label + 0.01) + (1 - pred) * np.log(1 - label + 0.01)  # 误差函数  **=^
        return s[0, 0]

    @classmethod
    def bottom_diff(self, pred, label):
        if label is None:
            return 0
        else:
            diff = np.zeros_like(pred)
            diff[0] = pred[0] - label
            return diff
    # 损失函数的导函数


class LstmParam(object):  # 网络初始化
    def __init__(self, mem_cell_ct, x_dim, y_dim=1, id=0):
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

        self.wg_cache = np.zeros_like(self.wg)
        self.wi_cache = np.zeros_like(self.wi)
        self.wf_cache = np.zeros_like(self.wf)
        self.wo_cache = np.zeros_like(self.wo)
        self.wh_cache = np.zeros_like(self.wh)
        self.wy_cache = np.zeros_like(self.wy)

        self.bg_cache = np.zeros_like(self.bg)
        self.bi_cache = np.zeros_like(self.bi)
        self.bf_cache = np.zeros_like(self.bf)
        self.bo_cache = np.zeros_like(self.bo)
        self.bh_cache = np.zeros_like(self.bh)
        self.by_cache = np.zeros_like(self.by)

    def apply_diff(self, lr=1):  # 梯度下降法修正误差
        # reset cache
        # print(lr)
        self.wg_cache = self.wg
        self.wi_cache = self.wi
        self.wf_cache = self.wf
        self.wo_cache = self.wo
        self.wh_cache = self.wh
        self.wy_cache = self.wy

        self.bg_cache = self.bg
        self.bi_cache = self.bi
        self.bf_cache = self.bf
        self.bo_cache = self.bo
        self.bh_cache = self.bh
        self.by_cache = self.by
        c = 1.0  # 正则化因子
        self.wg -= lr * self.wg_diff + lr * self.wg * c
        self.wi -= lr * self.wi_diff + lr * self.wi * c
        self.wf -= lr * self.wf_diff + lr * self.wf * c
        self.wo -= lr * self.wo_diff + lr * self.wo * c
        self.wh -= lr * self.wh_diff + lr * self.wh * c
        self.wy -= lr * self.wy_diff + lr * self.wy * c

        self.bg -= lr * self.bg_diff + lr * self.bg * c
        self.bi -= lr * self.bi_diff + lr * self.bi * c
        self.bf -= lr * self.bf_diff + lr * self.bf * c
        self.bo -= lr * self.bo_diff + lr * self.bo * c
        self.bh -= lr * self.bh_diff + lr * self.bh * c
        self.by -= lr * self.by_diff + lr * self.by * c
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

    def roll_back(self):
        self.wg = self.wg_cache
        self.wi = self.wi_cache
        self.wf = self.wf_cache
        self.wo = self.wo_cache
        self.wh = self.wh_cache
        self.wy = self.wy_cache

        self.bg = self.bg_cache
        self.bi = self.bi_cache
        self.bf = self.bf_cache
        self.bo = self.bo_cache
        self.bh = self.bh_cache
        self.by = self.by_cache

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


class memory(object):
    def __init__(self, cell_dim, y_dim, length):
        self.bg = np.zeros([cell_dim, length])
        self.bi = np.zeros([cell_dim, length])
        self.bf = np.zeros([cell_dim, length])
        self.bo = np.zeros([cell_dim, length])
        self.bh = np.zeros([cell_dim, length])
        self.by = np.zeros([y_dim, length])

    def param_memory(self, bg, bi, bf, bo, bh, by, i):  # 记录梯度
        self.bg[:, i] = list(bg)
        self.bi[:, i] = list(bi)
        self.bf[:, i] = list(bf)
        self.bh[:, i] = list(bh)
        self.bo[:, i] = list(bo)
        self.by[:, i] = list(by)

    def plt(self, A):
        for a in A:
            plt.plot(a)


class LstmState(object):
    def __init__(self, mem_cell_ct, x_dim, y_dim=1):  # 节点状态
        self.g = np.zeros([mem_cell_ct, 1])
        self.i = np.zeros([mem_cell_ct, 1])
        self.f = np.zeros([mem_cell_ct, 1])
        self.o = np.zeros([mem_cell_ct, 1])
        self.s = np.zeros([mem_cell_ct, 1])
        self.h = np.zeros([mem_cell_ct, 1])
        self.hh = np.zeros([mem_cell_ct, 1])
        self.y = np.zeros(y_dim)  # y_len
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)
        self.bottom_diff_x = np.zeros(x_dim)


class LstmNode(object):
    def __init__(self, lstm_param, lstm_state):
        # store reference to parameters and to activations
        self.state = lstm_state
        self.param = lstm_param  # net对象
        # non-recurrent input to node
        self.x = None
        # non-recurrent input concatenated with recurrent input
        self.xc = None
        self.sc = None
        self.h_prev = None
        self.s_prev = None

    def feed_forward(self, x, s_prev=None, h_prev=None):  # 前向计算 前馈
        # if this is the first lstm node in the network
        if s_prev is None:
            s_prev = np.zeros_like(self.state.s)
        if h_prev is None:
            h_prev = np.zeros_like(self.state.h)
        # save data for use in backprop
        self.s_prev = s_prev
        self.h_prev = h_prev

        # concatenate x(t) and h(t-1)
        xc = np.vstack((x,  h_prev))
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)  # input gate
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)  # forget gate
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)  # output gate
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        # s_t = new_message * input_gate + s_t-1 * forget_gate
        self.state.hh = np.tanh(np.dot(self.param.wh, self.state.s) + self.param.bh)
        self.state.h = self.state.hh * self.state.o
        self.state.y = sigmoid(np.dot(self.param.wy, self.state.h) + self.param.by)
        # cell output for gru model
        # self.state.h = self.state.s * self.state.o  # output
        self.x = x
        self.xc = xc
        self.sc = self.state.s
        return self.state.y

    def back_propagation(self, top_diff_h, top_diff_s, loss_y):  # 后向计算
        # notice that top_diff_s is carried along the constant error carousel
        # 注意 top_diff_表示是前N个神经元通过state传递下来的常数误差
        # self.state.o * top_diff_h 表示这个神经元通过链式求导传递进来的误差
        dh = self.param.wy.T * loss_y + top_diff_h
        dhh = dh * self.state.o
        ds = np.dot(self.param.wh.T, (1 - self.state.hh ** 2) * dhh) + top_diff_s
        do = dh * self.state.hh
        di = ds * self.state.g
        dg = ds * self.state.i
        df = ds * self.s_prev

        # diffs w.r.t. vector inside sigma / tanh function
        # diff(1/(1 + exp(-x))) = y(1 - y)
        di_input = (1. - self.state.i) * self.state.i * di
        df_input = (1. - self.state.f) * self.state.f * df
        do_input = (1. - self.state.o) * self.state.o * do
        # diff(tanh) = 1 - tanh^2 = 1 - y^2
        dg_input = (1. - self.state.g ** 2) * dg
        dh_input = (1. - self.state.hh ** 2) * dhh
        dy_input = loss_y

        # diffs w.r.t. inputs
        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.wh_diff += np.outer(dh_input, self.sc)
        self.param.wy_diff += np.outer(dy_input, self.state.h)

        self.param.bi_diff += di_input
        self.param.bf_diff += df_input
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input
        self.param.bh_diff += dh_input
        self.param.by_diff += dy_input

        # compute bottom diff
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # save bottom diffs
        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_x = dxc[:self.param.x_dim]
        self.state.bottom_diff_h = dxc[self.param.x_dim:]


class LstmNetwork(object):
    def __init__(self, y_list, x_list, cell_dim, y_dim=1, sampling=False):
        x_dim = len(x_list[0])
        self.x_dim = x_dim
        self.cell_dim = cell_dim
        self.y_dim = y_dim
        self.lstm_param = LstmParam(cell_dim, x_dim, y_dim, id=999)
        self.lstm_node_list = list()  # 节点(Node)
        # input sequence
        self.loop_id = 0
        self.x_list = x_list
        self.y_list = y_list
        self.y_predict = list()
        self.loop_num = len(y_list)
        self.lr = 0.42
        self.attenuation = 0.99
        self.max_loop = 400
        self.min_loss_diff = 0.000001
        self.start_time = 0
        self.sampling = sampling
        self.sample_x_len = 0
        self.sample_y_len = 0
        self.x_data = list()
        self.y_data = list()
        self.sample_num = 0
        self.cur_iter = 0
        self.sample_train_num = 50
        self.sample_flag = 0

    def train(self, lr=False, maxloop=False, min_loos_diff=False, attenuation=False,
              sample_x_len=False, sample_y_len=False, sample_train_num=False):
        assert len(self.y_list) == len(self.x_list)
        self.start_time = time.time()
        if lr:
            self.lr = lr
        if sample_train_num:
            self.sample_train_num = sample_train_num
        if not maxloop:
            maxloop = self.max_loop
        if not min_loos_diff:
            min_loos_diff = self.min_loss_diff
        if not attenuation:
            attenuation = self.attenuation
        if sample_y_len:
            self.sample_y_len = sample_y_len
        if sample_x_len:
            self.sample_x_len = sample_x_len
        lr_list = list()
        loss_list = list()
        loss_diff = 1
        loss = 0
        y_predict = list()
        lr_var = 1.0
        win_ratio = [0, ]
        x_list = self.x_list
        y_list = self.y_list
        if self.sampling:
            y_predict_list = [list() for _ in range(self.sample_train_num)]
            self.loop_num = self.sample_x_len
        else:
            y_predict_list = [list() for _ in self.y_list]
        self.net_initial()

        while self.cur_iter < maxloop and loss_diff > min_loos_diff:  # 迭代次数
            if self.sampling and self.sample_flag == 0 and self.cur_iter == 0:
                self.sample_initial(x_list, y_list)
            self.reset_loop()
            self.forward_loop()
            loss += self.backprop_loop()  # 计算误差并且计算梯度
            # lr = lr0 / (1 + np.exp(cur_iter*0.0085))

            if self.sampling:
                y_predict_list[self.sample_flag].append(self.y_predict[-1][0, 0])
                wr = self.sample_win_ratio_stat(self.y_data[self.sample_flag][-1], self.y_predict[-1][0, 0])
                win_ratio[self.cur_iter] += wr / self.sample_train_num
                self.sample_flag += 1
            else:
                for i in range(self.loop_num):
                    y_predict_list[i].append(self.y_predict[i][0, 0])
            if not self.sampling or self.sample_flag == self.sample_train_num:
                loss_list.append(loss)
                loss = 0
                self.sample_flag = 0
                print('cur_iter: ', self.cur_iter)
                self.lr *= attenuation
                if self.cur_iter > 0:
                    loss_diff = loss_list[self.cur_iter - 1] - loss_list[self.cur_iter]
                    if loss_diff < 0:
                        self.lstm_param.roll_back()
                        lr_var *= 0.4
                    else:
                        lr_list.append(self.lr)
                else:
                    lr_list.append(self.lr)
                    # lr = lr / self.sample_train_num mini-batch时的权重更新规则
                    # 也就是将100个样本的梯度求均值，替代online learning方法中单个样本的梯度值：
                self.cur_iter += 1
                win_ratio.append(0)
                self.lstm_param.apply_diff(lr=self.lr * lr_var / self.sample_train_num)  # 梯度下降法 修正net
            # if self.sampling and self.cur_iter % self.sample_train_num == 0:
                # print(self.y_list[-1])
                # print(self.y_predict[-1])

        for i in range(self.loop_num):
            y_predict.append(self.y_predict[i][0, 0])
        print("use time:", time.time() - self.start_time)
        print("lr:", lr)
        print("loss_diff: ", loss_diff)
        print("loss: ", loss_list[self.cur_iter - 1])
        loss_list[0] = loss_list[1]

        return loss_list, lr_list, y_predict_list, win_ratio

    def predict(self, x_list):
        self.net_initial(len(x_list))
        self.forward_loop(x_list=x_list)
        y_predict = list()
        for i in range(len(x_list)):
                y_predict.append(self.y_predict[i][0, 0])
        return y_predict

    def predict_sample(self, x_list):
        y_predict = list()
        for i in range(len(x_list) - self.sample_x_len):
            self.loop_id = 0
            self.y_predict = list()
            self.forward_loop(x_list=x_list[i: i+self.sample_x_len])
            y_predict.append(self.y_predict[-1][0, 0])
        return y_predict

    def net_initial(self):
        self.lstm_node_list = list()
        if self.sampling:
            self.loop_num = self.sample_x_len
            self.sample_num = len(self.x_list) - self.sample_x_len
            # self.sample_initial()
        for _ in range(self.loop_num):
            # need to add new lstm node, create new state mem
            lstm_state = LstmState(self.cell_dim, self.x_dim, self.y_dim)
            # 初始化状态参数
            self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))

    def reset_loop(self):
        self.y_predict = list()
        if self.sampling:
            self.x_list = self.x_data[self.sample_flag]
            self.y_list = self.y_data[self.sample_flag]

    def sample_win_ratio_stat(self, y1, y2):
        return (y1 - 0.5) * (y2 - 0.5) > 0

    def sample_initial(self, x_list, y_list):
        class_num = [0, 0]
        flag = 0
        self.x_data = list()
        self.y_data = list()
        while flag < self.sample_train_num:
            idx = int(np.random.rand() * self.sample_num)
            if y_list[idx + self.sample_x_len] == 1:
                if class_num[1] >= self.sample_train_num / 2:
                    continue
                class_num[1] += 1
            elif y_list[idx + self.sample_x_len] == 0:
                if class_num[0] >= self.sample_train_num / 2:
                    continue
                class_num[0] += 1
            else:
                continue
            x_sample = x_list[idx: idx + self.sample_x_len]
            y_sample = list()
            for j in range(self.sample_x_len):
                if j >= self.sample_x_len - self.sample_y_len:
                    y_sample.append(y_list[idx + j])
                else:
                    y_sample.append(None)
            self.x_data.append(x_sample)
            self.y_data.append(y_sample)
            flag += 1
        # print(class_num)

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
        loss = loss_layer.loss(self.y_predict[self.loop_id], self.y_list[self.loop_id])
        # 损失函数
        loss_y = loss_layer.bottom_diff(self.y_predict[self.loop_id], self.y_list[self.loop_id])
        # here s is not affecting loss due to h(t+1), hence we set equal to zero
        diff_h = np.zeros_like(self.lstm_param.bh)
        diff_s = np.zeros_like(self.lstm_param.bf)
        self.lstm_node_list[self.loop_id].back_propagation(diff_h, diff_s, loss_y)  # 修正net
        self.loop_id -= 1

        # ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        # we also propagate error along constant error carousel using diff_s
        # 按时间从后向前迭代
        while self.loop_id >= 0:
            loss += loss_layer.loss(self.y_predict[self.loop_id], self.y_list[self.loop_id])
            loss_y = loss_layer.bottom_diff(self.y_predict[self.loop_id], self.y_list[self.loop_id])
            diff_h += self.lstm_node_list[self.loop_id + 1].state.bottom_diff_h
            diff_s += self.lstm_node_list[self.loop_id + 1].state.bottom_diff_s
            self.lstm_node_list[self.loop_id].back_propagation(diff_h, diff_s, loss_y)  # 修正net
            self.loop_id -= 1
        self.loop_id = 0
        return loss

    def forward_loop(self, x_list=False):
        # get index of most recent x input
        # no recurrent inputs yet
        assert self.loop_id == 0
        if not x_list:
            x_list = self.x_list
            loop_num = self.loop_num
        else:
            loop_num = len(x_list)
        y = self.lstm_node_list[self.loop_id].feed_forward(x_list[self.loop_id])
        self.y_predict.append(y)
        while self.loop_id < loop_num - 1:
            s_prev = self.lstm_node_list[self.loop_id].state.s
            h_prev = self.lstm_node_list[self.loop_id].state.h
            self.loop_id += 1
            y = self.lstm_node_list[self.loop_id].feed_forward(x_list[self.loop_id], s_prev, h_prev)
            self.y_predict.append(y)
            # self.lstm_node_list[self.loop_id].feed_forward(x, s_prev, h_prev)


def sigmoid(x):  # 激活函数
    a = 1. / (1 + np.exp(-x))
    a[a > 0.99] = 0.99
    a[a < 0.01] = 0.01
    return a


# createst uniform random array w/ values in [a,b) and shape args
# 创建均匀分布[a, b]
# 当函数的参数不确定时，可以使用*args和**kwargs。*args没有key值，**kwargs有key值
# 修改为正态分布
def rand_arr(a, b, *args):
    np.random.seed(0)
    if False:
        c = np.random.rand(*args) * (b - a) + a
    else:
        c = np.random.randn(*args) * 0.05
    return c


def example():
    x_dim = 50
    y_list = [None, None, 0.1, 0.5, 0.8]
    np.random.seed(0)
    x_list = [np.random.random([x_dim, 1]) for _ in y_list]
    lstm_net = LstmNetwork(x_list=x_list, y_list=y_list, cell_dim=7)
    loss, lr, y_predict_list = lstm_net.train(lr=0.4, min_loos_diff=-999,
                                              attenuation=0.992, maxloop=400)
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
