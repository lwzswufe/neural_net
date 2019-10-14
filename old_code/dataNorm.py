# author='lwz'
# coding:utf-8
# !/usr/bin/env python3
# 数据归一化与标准化
import numpy as np


class DataNorm(object):
    def __init__(self):
        self.mean = list()
        self.weight = list()
        self.name = list()

    def norm(self, x, name):
        if name not in self.name:
            self.name.append(name)
            self.mean.append(np.mean(x))
            self.weight.append(np.std(x))
        flag = self.name.index(name)
        y = (x - self.mean[flag]) / self.weight[flag]
        return y

    def inverse_norm(self, x, name):
        # 逆向计算
        flag = self.name.index(name)
        y = (x * self.weight[flag]) + self.mean[flag]
        return y
