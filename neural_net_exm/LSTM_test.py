import numpy as np
import external_module
from layers import LSTMCell, LSTM
from torch_models import TorchLSTMCell
from losses import SquaredError
from optimizers import SGD


def err_fmt(params, golds, ix, warn_str=""):
    mine, label = params[ix]
    err_msg = "-" * 25 + " DEBUG " + "-" * 25 + "\n"
    prev_mine, prev_label = params[max(ix - 1, 0)]
    err_msg += "Mine (prev) [{}]:\n{}\n\nTheirs (prev) [{}]:\n{}".format(
        prev_label, prev_mine, prev_label, golds[prev_label]
    )
    err_msg += "\n\nMine [{}]:\n{}\n\nTheirs [{}]:\n{}".format(
        label, mine, label, golds[label]
    )
    err_msg += warn_str
    err_msg += "\n" + "-" * 23 + " END DEBUG " + "-" * 23
    return err_msg


def random_tensor(shape, standardize=False):
    """
    Create a random real-valued tensor of shape `shape`. If `standardize` is
    True, ensure each column has mean 0 and std 1.
    """
    offset = np.random.randint(-300, 300, shape)
    X = np.random.rand(*shape) + offset

    if standardize:
        eps = np.finfo(float).eps
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + eps)
    return X


def compare_LSTM(N=None):
    '''
    比较LSTM网络系统与pyTorch的梯度计算
    '''
    N = np.inf if N is None else N

    np.random.seed(12345)

    i = 1
    while i < N + 1:
        n_ex = np.random.randint(1, 10)
        n_in = np.random.randint(1, 10)
        n_out = np.random.randint(1, 10)
        n_t = np.random.randint(1, 10)
        X = random_tensor((n_ex, n_in, n_t), standardize=True)

        # initialize LSTM layer
        L1 = LSTMCell(n_out=n_out)

        # forward prop
        Cs = []
        y_preds = []
        for t in range(n_t):
            y_pred, Ct = L1.forward(X[:, :, t])
            y_preds.append(y_pred)
            Cs.append(Ct)

        # backprop
        dLdX = []
        dLdAt = np.ones_like(y_preds[t])
        for t in reversed(range(n_t)):
            dLdXt = L1.backward(dLdAt)
            dLdX.insert(0, dLdXt)
        dLdX = np.dstack(dLdX)
        y_preds = np.dstack(y_preds)
        Cs = np.array(Cs)

        # get gold standard gradients
        gold_mod = TorchLSTMCell(n_in, n_out, L1.parameters)
        golds = gold_mod.extract_grads(X)

        params = [
            (X, "X"),
            (np.array(Cs), "C"),
            (y_preds, "y"),
            (L1.parameters["bo"].T, "bo"),
            (L1.parameters["bu"].T, "bu"),
            (L1.parameters["bf"].T, "bf"),
            (L1.parameters["bc"].T, "bc"),
            (L1.parameters["Wo"], "Wo"),
            (L1.parameters["Wu"], "Wu"),
            (L1.parameters["Wf"], "Wf"),
            (L1.parameters["Wc"], "Wc"),
            (L1.gradients["bo"].T, "dLdBo"),
            (L1.gradients["bu"].T, "dLdBu"),
            (L1.gradients["bf"].T, "dLdBf"),
            (L1.gradients["bc"].T, "dLdBc"),
            (L1.gradients["Wo"], "dLdWo"),
            (L1.gradients["Wu"], "dLdWu"),
            (L1.gradients["Wf"], "dLdWf"),
            (L1.gradients["Wc"], "dLdWc"),
            (dLdX, "dLdX"),
        ]

        print("Case {}".format(i))
        for ix, (mine, label) in enumerate(params):
            np.testing.assert_allclose(
                mine,
                golds[label],
                err_msg=err_fmt(params, golds, ix),
                atol=1e-4,
                rtol=1e-4,
            )

            print("\tPASSED {}".format(label))
        i += 1


def test_LSTM():
    '''
    测试LSTM
    '''
    n_ex = 4   # 样本数
    n_in = 25  # 输入数据维度
    n_out = 1  # 输出数据维度
    n_t = 5    # 时间周期数
    np.random.seed(0)

    y = np.random.random((n_ex, n_out, n_t))
    X = random_tensor((n_ex, n_in, n_t), standardize=True)

    # initialize net layer
    net = LSTM(n_in, n_out, n_t, optimizer=SGD())
    loss_cls = SquaredError()
    for i in range(1000):
        y_pred = net.forward(X)
        loss = loss_cls(y, y_pred)
        Z = np.dstack(net.derived_variables["Go"])
        grad = loss_cls.grad(y, y_pred, Z, net.act_fn)
        net.backward(grad)
        net.update()
        print("iter_{:03d} loss:{:.4f}".format(i + 1, np.sum(loss)))


if __name__ == "__main__":
    # compare_LSTM(100)
    test_LSTM()
