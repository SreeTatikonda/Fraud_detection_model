"""
Anomaly detection branch for ATO fraud detection.
Two complementary unsupervised models:
  1. Isolation Forest  — tree-based, good at global outliers
  2. Autoencoder       — learns normal behaviour; high reconstruction = anomalous

Both operate on the 35-dim feature vector. Scores are averaged (weighted).
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Autoencoder (pure NumPy — no torch required)
# ---------------------------------------------------------------------------

class NumpyAutoencoder:
    """
    Shallow autoencoder: 35 → 16 → 8 → 16 → 35
    Uses ReLU activations. Trained with MSE loss via mini-batch SGD.
    """

    def __init__(self, input_dim: int = 35, hidden1: int = 16, bottleneck: int = 8,
                 lr: float = 1e-3, epochs: int = 100, batch_size: int = 256,
                 seed: int = 42):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        def init(fan_in, fan_out):
            s = np.sqrt(2.0 / fan_in)
            return (rng.normal(0, s, (fan_out, fan_in)).astype(np.float32),
                    np.zeros(fan_out, dtype=np.float32))

        self.W1, self.b1 = init(input_dim, hidden1)
        self.W2, self.b2 = init(hidden1, bottleneck)
        self.W3, self.b3 = init(bottleneck, hidden1)
        self.W4, self.b4 = init(hidden1, input_dim)
        self.scaler = StandardScaler()
        self.threshold: Optional[float] = None

    @staticmethod
    def _relu(x): return np.maximum(x, 0)

    @staticmethod
    def _relu_grad(x): return (x > 0).astype(np.float32)

    def _forward(self, X):
        z1 = X @ self.W1.T + self.b1;  a1 = self._relu(z1)
        z2 = a1 @ self.W2.T + self.b2; a2 = self._relu(z2)
        z3 = a2 @ self.W3.T + self.b3; a3 = self._relu(z3)
        z4 = a3 @ self.W4.T + self.b4  # linear output
        return z4, (z1, a1, z2, a2, z3, a3)

    def fit(self, X_raw: np.ndarray) -> "NumpyAutoencoder":
        X = self.scaler.fit_transform(X_raw).astype(np.float32)
        N = len(X)
        for epoch in range(self.epochs):
            idx = np.random.permutation(N)
            epoch_loss = 0.0
            for start in range(0, N, self.batch_size):
                batch = X[idx[start:start + self.batch_size]]
                out, (z1, a1, z2, a2, z3, a3) = self._forward(batch)
                diff = out - batch                         # (B, D)
                loss = (diff ** 2).mean()
                epoch_loss += loss

                # Backprop
                dout = 2 * diff / batch.shape[0]
                dW4 = dout.T @ a3;        db4 = dout.sum(axis=0)
                da3 = dout @ self.W4
                dz3 = da3 * self._relu_grad(z3)
                dW3 = dz3.T @ a2;         db3 = dz3.sum(axis=0)
                da2 = dz3 @ self.W3
                dz2 = da2 * self._relu_grad(z2)
                dW2 = dz2.T @ a1;         db2 = dz2.sum(axis=0)
                da1 = dz2 @ self.W2
                dz1 = da1 * self._relu_grad(z1)
                dW1 = dz1.T @ batch;      db1 = dz1.sum(axis=0)

                for param, grad in [(self.W4, dW4), (self.b4, db4),
                                    (self.W3, dW3), (self.b3, db3),
                                    (self.W2, dW2), (self.b2, db2),
                                    (self.W1, dW1), (self.b1, db1)]:
                    param -= self.lr * grad

            if epoch % 20 == 0:
                print(f"  AE epoch {epoch:3d}  loss={epoch_loss:.4f}")

        # Set threshold at 95th percentile of training reconstruction errors
        out, _ = self._forward(X)
        errors = ((out - X) ** 2).mean(axis=1)
        self.threshold = float(np.percentile(errors, 95))
        return self

    def reconstruction_error(self, X_raw: np.ndarray) -> np.ndarray:
        X = self.scaler.transform(X_raw).astype(np.float32)
        out, _ = self._forward(X)
        return ((out - X) ** 2).mean(axis=1)

    def score(self, X_raw: np.ndarray) -> np.ndarray:
        """Returns anomaly score in [0, 1]. Higher = more anomalous."""
        errors = self.reconstruction_error(X_raw)
        if self.threshold and self.threshold > 0:
            return np.clip(errors / (self.threshold * 3), 0, 1)
        return np.clip(errors / (errors.max() + 1e-8), 0, 1)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "NumpyAutoencoder":
        with open(path, 'rb') as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Isolation Forest wrapper
# ---------------------------------------------------------------------------

class AnomalyIsolationForest:
    """
    Wraps sklearn IsolationForest with calibrated [0,1] output.
    contamination should match expected fraud rate (often 0.005–0.02).
    """

    def __init__(self, n_estimators: int = 200, contamination: float = 0.01,
                 random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=0.8,
            bootstrap=True,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.trained = False

    def fit(self, X_raw: np.ndarray) -> "AnomalyIsolationForest":
        X = self.scaler.fit_transform(X_raw)
        self.model.fit(X)
        self.trained = True
        return self

    def score(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Returns anomaly score in [0, 1].
        sklearn decision_function returns negative scores for anomalies —
        we flip and normalise.
        """
        X = self.scaler.transform(X_raw)
        raw = self.model.decision_function(X)   # negative = anomalous
        # Clip to known range and normalise to [0, 1]
        clipped = np.clip(-raw, 0, None)
        norm = clipped / (clipped.max() + 1e-3)
        return np.clip(norm, 0, 1).astype(np.float32)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "AnomalyIsolationForest":
        with open(path, 'rb') as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Combined anomaly scorer
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Combines IsolationForest (weight=0.4) and Autoencoder (weight=0.6).
    The AE gets higher weight because it captures sequential session context.
    """

    def __init__(self, if_weight: float = 0.4, ae_weight: float = 0.6):
        self.iso_forest = AnomalyIsolationForest()
        self.autoencoder = NumpyAutoencoder()
        self.if_weight = if_weight
        self.ae_weight = ae_weight

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        print("Training Isolation Forest...")
        self.iso_forest.fit(X)
        print("Training Autoencoder...")
        self.autoencoder.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Returns combined anomaly score in [0, 1]."""
        if_scores = self.iso_forest.score(X)
        ae_scores  = self.autoencoder.score(X)
        return self.if_weight * if_scores + self.ae_weight * ae_scores

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "AnomalyDetector":
        with open(path, 'rb') as f:
            return pickle.load(f)
