import numpy as np
import external_module
from rnn_sequential_sample import RNN_SeqSample
from torch_models import TorchRNNCell
from losses import SquaredError
from optimizers import SGD


def test_RNN():
    n_ex = 4   # 样本数
    n_in = 25  # 输入数据维度
    n_out = 1  # 输出数据维度
    n_t = 5    # 时间周期数
    np.random.seed(0)

    y = np.random.random((n_ex, n_out, n_t))
    X = random_tensor((n_ex, n_in, n_t), standardize=True)

    # initialize RNN layer
    rnn = RNN_SeqSample(n_in, n_out, n_t, optimizer=SGD())
    loss_cls = SquaredError()
    for i in range(1000):
        y_pred = rnn.forward(X)
        loss = loss_cls(y, y_pred)
        Z = np.dstack(rnn.derived_variables["Z"])
        grad = loss_cls.grad(y, y_pred, Z, rnn.act_fn)
        rnn.backward(grad)
        rnn.update()
        print("iter_{:03d} loss:{:.4f}".format(i + 1, np.sum(loss)))


if __name__ == "__main__":
    # compare_RNNCell(100)
    test_RNN()
