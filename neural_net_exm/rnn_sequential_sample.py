import numpy as np
import external_module
from layers import RNN


class RNN_SeqSample(RNN):
    def forward(self, X):
        """
        Run a forward pass across all timesteps in the input.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex + n_t - 1, n_in)`
            Input consisting of `n_ex` examples each of dimensionality `n_in`
            and extending for `n_t` timesteps.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex + n_t - 2, n_out)`
            The value of the hidden state for each of the `n_ex` examples
            across each of the `n_t` timesteps.
        """
        if not self.is_initialized:
            self.n_in = X.shape[1]
            self._init_params()

        Y = []
        x_len, n_in = X.shape
        for t in range(self.n_timesteps):
            idx_ed = x_len + t - self.n_timesteps
            yt = self.cell.forward(X[t:idx_ed, :])
            Y.append(yt)
        return np.dstack(Y)
