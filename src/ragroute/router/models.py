"""Quality models for the pairwise router: predict quality in [0,1] for a (query, config) pair.

  lr   : fractional logistic regression. Soft targets y in [0,1] are fitted by duplicating each row as a
         positive with weight y and a negative with weight 1-y (exact cross-entropy with soft labels).
         Uses explicit query x config interaction features (otherwise it could only learn config main
         effects and every query would get the same config ranking).
  gbdt : LightGBM with objective='cross_entropy' (accepts labels in [0,1]); learns interactions itself.
         Falls back to sklearn HistGradientBoostingRegressor if LightGBM is not installed.
  mlp  : sklearn MLPRegressor (squared loss), outputs clipped to [0,1].
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class FractionalLogit:
    needs_interactions = True

    def __init__(self, seed: int = 0, C: float = 0.5):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=C, max_iter=3000, random_state=seed)

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        y = np.clip(np.asarray(y, float), 0, 1)
        Xs = np.vstack([X, X])
        ys = np.r_[np.ones(len(y)), np.zeros(len(y))]
        w = np.r_[y, 1 - y]
        keep = w > 0
        self.model.fit(Xs[keep], ys[keep], sample_weight=w[keep])
        return self

    def predict(self, X):
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]


class GBDT:
    needs_interactions = False

    def __init__(self, seed: int = 0, n_estimators: int = 400, learning_rate: float = 0.05, num_leaves: int = 31):
        try:
            import lightgbm as lgb

            self.model = lgb.LGBMRegressor(objective="cross_entropy", n_estimators=n_estimators,
                                           learning_rate=learning_rate, num_leaves=num_leaves, min_child_samples=40,
                                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                           random_state=seed, verbose=-1)
            self.backend = "lightgbm"
        except ImportError:  # pragma: no cover
            from sklearn.ensemble import HistGradientBoostingRegressor

            self.model = HistGradientBoostingRegressor(max_iter=n_estimators, learning_rate=learning_rate,
                                                       random_state=seed)
            self.backend = "sklearn_hgb"

    def fit(self, X, y):
        self.model.fit(np.asarray(X), np.clip(np.asarray(y, float), 0, 1))
        return self

    def predict(self, X):
        return np.clip(self.model.predict(np.asarray(X)), 0, 1)

    def feature_importance(self):
        if self.backend == "lightgbm":
            return self.model.booster_.feature_importance(importance_type="gain")
        return None


class MLP:
    needs_interactions = True

    def __init__(self, seed: int = 0):
        self.scaler = StandardScaler()
        self.model = MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True,
                                  validation_fraction=0.1, max_iter=300, random_state=seed)

    def fit(self, X, y):
        self.model.fit(self.scaler.fit_transform(X), np.asarray(y, float))
        return self

    def predict(self, X):
        return np.clip(self.model.predict(self.scaler.transform(X)), 0, 1)


MODELS = {"lr": FractionalLogit, "gbdt": GBDT, "mlp": MLP}


def make_model(kind: str, seed: int):
    return MODELS[kind](seed=seed)
