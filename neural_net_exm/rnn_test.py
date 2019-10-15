import numpy as np
import external_module
from layers import RNNCell, RNN
from torch_models import TorchRNNCell
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


def compare_RNNCell(N=None):
    N = np.inf if N is None else N

    np.random.seed(12345)

    i = 1
    while i < N + 1:
        n_ex = np.random.randint(1, 10)
        n_in = np.random.randint(1, 10)
        n_out = np.random.randint(1, 10)
        n_t = np.random.randint(1, 10)
        X = random_tensor((n_ex, n_in, n_t), standardize=True)

        # initialize RNN layer
        L1 = RNNCell(n_out=n_out)

        # forward prop
        y_preds = []
        for t in range(n_t):
            y_pred = L1.forward(X[:, :, t])
            y_preds += [y_pred]

        # backprop
        dLdX = []
        dLdAt = np.ones_like(y_preds[t])
        for t in reversed(range(n_t)):
            dLdXt = L1.backward(dLdAt)
            dLdX.insert(0, dLdXt)
        dLdX = np.dstack(dLdX)

        # get gold standard gradients
        gold_mod = TorchRNNCell(n_in, n_out, L1.parameters)
        golds = gold_mod.extract_grads(X)

        params = [
            (X, "X"),
            (np.array(y_preds), "y"),
            (L1.parameters["ba"].T, "ba"),
            (L1.parameters["bx"].T, "bx"),
            (L1.parameters["Wax"].T, "Wax"),
            (L1.parameters["Waa"].T, "Waa"),
            (L1.gradients["ba"].T, "dLdBa"),
            (L1.gradients["bx"].T, "dLdBx"),
            (L1.gradients["Wax"].T, "dLdWax"),
            (L1.gradients["Waa"].T, "dLdWaa"),
            (dLdX, "dLdX"),
        ]

        print("Trial {}".format(i))
        for ix, (mine, label) in enumerate(params):
            np.testing.assert_allclose(
                mine,
                golds[label],
                err_msg=err_fmt(params, golds, ix),
                atol=1e-3,
                rtol=1e-3,
            )
            print("\tPASSED {}".format(label))
        i += 1


def test_RNN():
    n_ex = 4   # 样本数
    n_in = 25  # 输入数据维度
    n_out = 1  # 输出数据维度
    n_t = 5    # 时间周期数
    np.random.seed(0)

    y = np.random.random((n_ex, n_out, n_t))
    X = random_tensor((n_ex, n_in, n_t), standardize=True)

    # initialize RNN layer
    rnn = RNN(n_in, n_out, n_t, optimizer=SGD())
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
