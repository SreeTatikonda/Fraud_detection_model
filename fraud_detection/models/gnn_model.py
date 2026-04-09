"""
Graph Neural Network (GNN) for ATO fraud detection.
Uses GraphSAGE to aggregate 2-hop neighborhood embeddings over the
account-device-IP graph. Accounts sharing devices/IPs with known fraudsters
get elevated risk embeddings.

Requirements: torch, torch_geometric
Install: pip install torch torch_geometric
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Pure-numpy GraphSAGE (no torch dependency — swap for PyG in production)
# This version is fully self-contained and testable without a GPU.
# ---------------------------------------------------------------------------

class GraphSAGELayer:
    """
    One GraphSAGE aggregation step.
    out_i = ReLU(W_self @ h_i + W_neigh @ mean(h_neighbors))
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        self.W_self  = rng.normal(0, scale, (out_dim, in_dim)).astype(np.float32)
        self.W_neigh = rng.normal(0, scale, (out_dim, in_dim)).astype(np.float32)
        self.bias    = np.zeros(out_dim, dtype=np.float32)

    def forward(self, h: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        h   : (N, in_dim)  node features
        adj : (N, N)       row-normalised adjacency (no self-loops)
        """
        agg = adj @ h                                    # (N, in_dim) — mean neighbor
        out = h @ self.W_self.T + agg @ self.W_neigh.T + self.bias
        return np.maximum(out, 0)                        # ReLU


class FraudGNN:
    """
    2-layer GraphSAGE followed by a sigmoid output head.
    Input:  node feature matrix (N x 35), adjacency matrix (N x N)
    Output: per-node fraud probability (N,)
    """

    def __init__(self, feat_dim: int = 35, hidden: int = 64, out_dim: int = 32):
        self.layer1 = GraphSAGELayer(feat_dim, hidden, seed=0)
        self.layer2 = GraphSAGELayer(hidden, out_dim, seed=1)
        rng = np.random.default_rng(2)
        self.W_out = rng.normal(0, 0.1, (1, out_dim)).astype(np.float32)
        self.b_out = np.zeros(1, dtype=np.float32)
        self.trained = False

    def _normalise_adj(self, A: np.ndarray) -> np.ndarray:
        """Row-normalise adjacency (no self-loops)."""
        np.fill_diagonal(A, 0)
        row_sum = A.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1, row_sum)
        return A / row_sum

    def forward(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Returns fraud probability for each node."""
        A_norm = self._normalise_adj(A.copy())
        h1 = self.layer1.forward(X, A_norm)
        h2 = self.layer2.forward(h1, A_norm)
        logits = (h2 @ self.W_out.T).squeeze() + self.b_out
        return 1.0 / (1.0 + np.exp(-logits))   # sigmoid

    def fit(self, X: np.ndarray, A: np.ndarray, labels: np.ndarray,
            lr: float = 0.01, epochs: int = 50) -> "FraudGNN":
        """
        Simplified gradient descent training.
        In production: use PyTorch Geometric with mini-batch sampling.
        """
        for epoch in range(epochs):
            probs = self.forward(X, A)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

            # Gradient of output layer (simplified backprop)
            dout = (probs - labels) / len(labels)
            A_norm = self._normalise_adj(A.copy())
            h1 = self.layer1.forward(X, A_norm)
            h2 = self.layer2.forward(h1, A_norm)
            grad_W_out = (dout[:, None] * h2).mean(axis=0, keepdims=True)
            self.W_out -= lr * grad_W_out
            self.b_out -= lr * dout.mean()

            if epoch % 10 == 0:
                print(f"  GNN epoch {epoch:3d}  loss={loss:.4f}")

        self.trained = True
        return self

    def score(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Return fraud probabilities (alias for forward)."""
        return self.forward(X, A)


# ---------------------------------------------------------------------------
# Graph builder utility
# ---------------------------------------------------------------------------

class AccountGraph:
    """
    Builds the N x N adjacency matrix and N x feat_dim feature matrix
    from a list of SessionEvent-like dicts for GNN consumption.
    """

    def __init__(self, feat_dim: int = 35):
        self.feat_dim = feat_dim
        self.account_index: Dict[str, int] = {}
        self.features: List[np.ndarray] = []
        self.edges: List[Tuple[int, int, float]] = []

    def add_account(self, account_id: str, feature_vector: np.ndarray) -> int:
        if account_id not in self.account_index:
            idx = len(self.account_index)
            self.account_index[account_id] = idx
            self.features.append(feature_vector[:self.feat_dim])
        return self.account_index[account_id]

    def add_edge(self, acc_a: str, acc_b: str, weight: float = 1.0):
        if acc_a in self.account_index and acc_b in self.account_index:
            i, j = self.account_index[acc_a], self.account_index[acc_b]
            self.edges.append((i, j, weight))
            self.edges.append((j, i, weight))

    def to_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (X, A) ready for FraudGNN.forward()."""
        N = len(self.account_index)
        X = np.stack(self.features).astype(np.float32)
        A = np.zeros((N, N), dtype=np.float32)
        for i, j, w in self.edges:
            A[i, j] = w
        return X, A

    def get_scores(self, model: FraudGNN) -> Dict[str, float]:
        """Returns {account_id: fraud_prob} mapping."""
        X, A = self.to_matrices()
        probs = model.score(X, A)
        if probs.ndim == 0:
            probs = np.array([float(probs)])
        return {acc: float(probs[idx])
                for acc, idx in self.account_index.items()}


# ---------------------------------------------------------------------------
# PyTorch Geometric version (production — uncomment when torch_geometric is available)
# ---------------------------------------------------------------------------

PYTORCH_GEO_CODE = '''
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

class FraudGNNTorch(torch.nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 64, out_dim: int = 32):
        super().__init__()
        self.conv1 = SAGEConv(feat_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)
        self.head  = torch.nn.Linear(out_dim, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return torch.sigmoid(self.head(h)).squeeze()

# Training loop:
# data = Data(x=X_tensor, edge_index=edge_index, y=labels)
# model = FraudGNNTorch(feat_dim=35)
# opt = torch.optim.Adam(model.parameters(), lr=1e-3)
# for epoch in range(200):
#     model.train()
#     opt.zero_grad()
#     out = model(data.x, data.edge_index)
#     loss = F.binary_cross_entropy(out[train_mask], data.y[train_mask].float())
#     loss.backward(); opt.step()
'''
