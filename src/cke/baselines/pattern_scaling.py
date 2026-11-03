"""Pattern scaling: global-mean T from global forcings, then local field = a + b * T_global per cell."""
import numpy as np
from sklearn.linear_model import LinearRegression


class PatternScaling:
    def __init__(self):
        self.global_model = LinearRegression()
        self.a = None
        self.b = None

    def fit(self, global_forcing: np.ndarray, target: np.ndarray, tglobal: np.ndarray):
        """global_forcing (T, F); target (T, H, W); tglobal (T,) global-mean temperature."""
        self.global_model.fit(global_forcing, tglobal)
        T = target.shape[0]
        A = np.stack([np.ones(T), tglobal], 1)
        coef, *_ = np.linalg.lstsq(A, target.reshape(T, -1), rcond=None)
        self.a = coef[0].reshape(target.shape[1:])
        self.b = coef[1].reshape(target.shape[1:])
        return self

    def predict_global(self, global_forcing):
        return self.global_model.predict(global_forcing)

    def predict(self, global_forcing):
        tg = self.predict_global(global_forcing)
        return self.a[None] + self.b[None] * tg[:, None, None]
