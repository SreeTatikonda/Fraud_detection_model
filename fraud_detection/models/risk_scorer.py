"""
Final risk scorer: combines GNN, anomaly, and supervised ensemble scores
via a meta-logistic regression, then routes to a risk decision engine.

Decision engine outputs one of three actions:
  ALLOW       — score < low_threshold
  STEP_UP_MFA — low_threshold <= score < high_threshold
  BLOCK       — score >= high_threshold
"""

from __future__ import annotations
import pickle
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Risk action enum
# ---------------------------------------------------------------------------

class RiskAction(str, Enum):
    ALLOW       = "ALLOW"
    STEP_UP_MFA = "STEP_UP_MFA"
    BLOCK       = "BLOCK"


@dataclass
class RiskDecision:
    account_id: str
    event_id: str
    risk_score: float           # 0.0 – 1.0
    action: RiskAction
    gnn_score: float
    anomaly_score: float
    ensemble_score: float
    latency_ms: float
    explanation: str


# ---------------------------------------------------------------------------
# Final risk meta-scorer
# ---------------------------------------------------------------------------

class RiskMetaScorer:
    """
    Combines three model scores into one final risk probability.
    Trained on labeled events where each score has already been computed.

    Input features (per event):
      [gnn_score, anomaly_score, ensemble_score,
       gnn_x_anomaly,             # interaction term
       ensemble_x_anomaly]        # interaction term

    The interaction terms help when multiple signals agree (multiplicative risk).
    """

    def __init__(self, gnn_weight: float = 0.3,
                 anomaly_weight: float = 0.25,
                 ensemble_weight: float = 0.45):
        # Fallback weights (used before calibration)
        self.gnn_weight      = gnn_weight
        self.anomaly_weight  = anomaly_weight
        self.ensemble_weight = ensemble_weight

        self.scaler = StandardScaler()
        self.lr = LogisticRegression(C=2.0, max_iter=1000,
                                     class_weight='balanced',
                                     random_state=42)
        self.calibrated = False

    def _build_meta_features(self, gnn: np.ndarray, anomaly: np.ndarray,
                               ensemble: np.ndarray) -> np.ndarray:
        return np.column_stack([
            gnn, anomaly, ensemble,
            gnn * anomaly,
            ensemble * anomaly,
        ])

    def fit(self, gnn_scores: np.ndarray, anomaly_scores: np.ndarray,
            ensemble_scores: np.ndarray, labels: np.ndarray) -> "RiskMetaScorer":
        X = self._build_meta_features(gnn_scores, anomaly_scores, ensemble_scores)
        X_scaled = self.scaler.fit_transform(X)
        self.lr.fit(X_scaled, labels)
        self.calibrated = True
        return self

    def score(self, gnn: float, anomaly: float, ensemble: float) -> float:
        """Returns final fraud probability [0, 1]."""
        if self.calibrated:
            X = self._build_meta_features(
                np.array([gnn]), np.array([anomaly]), np.array([ensemble])
            )
            X_scaled = self.scaler.transform(X)
            return float(self.lr.predict_proba(X_scaled)[0, 1])
        else:
            # Weighted average fallback
            return (self.gnn_weight * gnn +
                    self.anomaly_weight * anomaly +
                    self.ensemble_weight * ensemble)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "RiskMetaScorer":
        with open(path, 'rb') as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Risk decision engine
# ---------------------------------------------------------------------------

class RiskDecisionEngine:
    """
    Thresholded routing on the final risk score.

    Thresholds should be tuned on a validation set using precision/recall
    trade-offs aligned with business cost matrix (false negative >> false positive).

    low_threshold:  below this → ALLOW
    high_threshold: above this → BLOCK
    between:        STEP_UP_MFA (challenge with 2FA)
    """

    def __init__(self,
                 low_threshold: float = 0.3,
                 high_threshold: float = 0.7,
                 meta_scorer: Optional[RiskMetaScorer] = None):
        self.low_threshold  = low_threshold
        self.high_threshold = high_threshold
        self.meta_scorer = meta_scorer or RiskMetaScorer()

    def decide(self, account_id: str, event_id: str,
               gnn_score: float, anomaly_score: float,
               ensemble_score: float) -> RiskDecision:
        t0 = time.perf_counter()

        risk_score = self.meta_scorer.score(gnn_score, anomaly_score, ensemble_score)

        if risk_score < self.low_threshold:
            action = RiskAction.ALLOW
        elif risk_score < self.high_threshold:
            action = RiskAction.STEP_UP_MFA
        else:
            action = RiskAction.BLOCK

        # Human-readable explanation (highest contributing signal)
        scores = {'GNN graph risk': gnn_score,
                  'Behavioral anomaly': anomaly_score,
                  'Supervised model': ensemble_score}
        top_signal = max(scores, key=scores.__getitem__)
        explanation = (f"Action={action.value}  score={risk_score:.3f}  "
                       f"primary_signal={top_signal}({scores[top_signal]:.3f})")

        latency_ms = (time.perf_counter() - t0) * 1000

        return RiskDecision(
            account_id=account_id,
            event_id=event_id,
            risk_score=round(risk_score, 4),
            action=action,
            gnn_score=round(gnn_score, 4),
            anomaly_score=round(anomaly_score, 4),
            ensemble_score=round(ensemble_score, 4),
            latency_ms=round(latency_ms, 3),
            explanation=explanation,
        )

    def update_thresholds(self, low: float, high: float):
        """Hot-reload thresholds without redeploying the model."""
        assert 0 < low < high < 1, "Thresholds must satisfy 0 < low < high < 1"
        self.low_threshold  = low
        self.high_threshold = high

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "RiskDecisionEngine":
        with open(path, 'rb') as f:
            return pickle.load(f)
