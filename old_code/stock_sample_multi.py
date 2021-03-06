# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
import pandas as pd
import numpy as np
from lstm_multi import LstmNetwork
import matplotlib.pyplot as plt
import dataNorm
global data_norm


def load_data(stock='SH600000'):
    # fname = 'D:\\data\\temp\\' + stock + '.csv'
    fname = 'D:\\data\\minute_csv\\' + stock + '.csv'
    df = pd.read_csv(fname)
    df = df[df['vol'] > 0]
    return df[700:]


def data_processing(df):
    global data_norm
    x_list = list()
    y_list = list()
    close = np.array(df.close)
    high = np.array(df.high)
    low = np.array(df.low)
    vol = np.array(df.vol)
    atr = np.log(high) - np.log(low)
    ret_5 = np.log(np.array(close[5:])) - np.log(np.array(close[:-5]))
    ret_1 = np.log(np.array(close[10:])) - np.log(np.array(close[5:-5]))

    ret_y = ret_1 * 3 + 0.5
    c = 0.000
    ret_1[ret_1 >= c] = 1
    ret_1[(ret_1 < c) & (ret_1 >= 0)] = 1
    ret_1[(ret_1 < 0) & (ret_1 > -c)] = 1
    ret_1[ret_1 <= -c] = 0
    for i in range(int(max(ret_1)) + 1):
        print('class_' + str(i+1) + '_sample_num:' + str(sum(ret_1 == i)))

    ma_5 = ma(close, 5) / close[5:] - 1
    atr_ma_5 = ma(atr, 5) / close[5:] - 1
    vol = np.log(vol[5:])
    atr = atr[5:]

    atr = data_norm.norm(atr, 'atr')
    ma_5 = data_norm.norm(ma_5, 'ma_5')
    ret_5 = data_norm.norm(ret_5, 'ret_5')
    vol = data_norm.norm(vol, 'vol')
    atr_ma_5 = data_norm.norm(atr_ma_5, 'atr_ma_5')

    for i in range(len(atr) - 5):
        y_list.append(ret_1[i])
        x_list.append(np.array([[atr[i]], [ma_5[i]], [ret_5[i]], [vol[i]], [atr_ma_5[i]]]))
    return x_list, y_list, ret_y


def normalization(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x


def ma(array_in, length):
    num = len(array_in)
    ma_n = np.zeros(num)
    for i in range(length):
        try:
            ma_n[i:] += array_in[:num-i]
        except:
            print('here')
    ma_n /= length
    return ma_n[length:]


def win_ratio_stat(y_predict_list, yy):
    ratio = list()
    N = len(yy)
    for i in range(len(y_predict_list[0])):
        ratio.append(0)
        for j in range(N):
            ratio[i] += ((y_predict_list[j][i] - 0.5) * (yy[j][-1] - 0.5) > 0)
    return ratio


def trade_test(df, y_predict):
    df.index = range(len(df))
    asset = [100, ]
    position = 0 + (y_predict[0] > 0.55)
    for i in range(1, len(y_predict)):
        asset.append((1 - position) * asset[i - 1] +
                     position * asset[i - 1] * df.close[i] / df.close[i - 1])
        if y_predict[i] > 0.55:
            position = 1
        else:
            position = 0
    return asset


if __name__ == "__main__":
    df = load_data()
    # np.random.seed(0)
    data_norm = dataNorm.DataNorm()
    x_list, y_list, yy = data_processing(df[:1810])

    lstm_net = LstmNetwork(x_list=x_list, y_list=y_list, cell_dim=40, sampling=True)
    loss, lr, err = lstm_net.train(lr=0.01, min_loos_diff=-999,
                                   attenuation=0.990, maxloop=400,
                                   sample_x_len=10, sample_y_len=5,
                                   sample_train_num=60)
    loss = np.array(loss)
    t = 3610
    err.plot(1)
    err.plot(2)
    x_list2, y_list2, yy = data_processing(df[:t])
    # y_predict = lstm_net.predict(x_list2)
    y_predict = lstm_net.predict_sample(x_list2)
    # asset = trade_test(df[15:t-5], y_predict)
    df = df[15:t-5]
    df['state'] = np.array(y_predict)
    df.date = pd.to_datetime(df.date)
    df.index = range(len(df))
    if True:
        plt.plot(df.close, color='black', lw=1, label='hs300')
        plt.plot(df.close[df.state == 2], 'o', color='red', lw=1, label='up')
        plt.plot(df.close[df.state == 1], 'o', color='yellow', lw=1, label='mid_state')
        plt.plot(df.close[df.state == 0], 'o', color='green', lw=1, label='down')
    else:
        plt.plot(df.date, df.close, color='black', lw=1, label='SH600000')
        plt.plot(df.date[df.state == 2], df.close[df.state == 2], 'o', color='red', lw=1, label='up_state')
        plt.plot(df.date[df.state == 1], df.close[df.state == 1], 'o', color='yellow', lw=1, label='mid_state')
    # plt.plot(df.date[df.state == 1], df.close[df.state == 1], 'o', color='blue', lw=1, label='mid_state')
        plt.plot(df.date[df.state == 0], df.close[df.state == 0], 'o', color='green', lw=1, label='down_state')
    plt.legend()
    plt.show()
    # ratio = win_ratio_stat(y_predict_list, y_list)
    # plt.plot(wr)
    # plt.show()
    # plt.plot(asset)
    # plt.show()
    plt.plot(loss[:-1] - loss[1:])
    plt.show()
    plt.plot(loss)
    legend = ['loss']
    plt.legend(legend)
    plt.show()
    i = 0
    # for y in y_predict_list:
    #     plt.plot(y, '.')
    #     i += 1
    #    legend.append('y_' + str(i))