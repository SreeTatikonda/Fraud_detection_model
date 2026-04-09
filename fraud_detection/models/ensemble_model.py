"""
Supervised ensemble for ATO fraud detection.
XGBoost + LightGBM trained on labeled data, combined via a stacking
meta-learner (logistic regression on out-of-fold predictions).

Handles severe class imbalance (typical fraud rate: 0.1–2%).
"""

from __future__ import annotations
import pickle
import warnings
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# XGBoost wrapper
# ---------------------------------------------------------------------------

class XGBoostFraudModel:
    def __init__(self, n_estimators: int = 500, max_depth: int = 6,
                 learning_rate: float = 0.05, scale_pos_weight: float = 50.0,
                 subsample: float = 0.8, colsample_bytree: float = 0.8,
                 random_state: int = 42):
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=scale_pos_weight,   # handles imbalance
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                eval_metric='aucpr',
                use_label_encoder=False,
                random_state=random_state,
                n_jobs=-1,
            )
            self._available = True
        except ImportError:
            print("xgboost not installed — using sklearn GradientBoostingClassifier fallback")
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=random_state
            )
            self._available = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "XGBoostFraudModel":
        if self._available and X_val is not None:
            self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# LightGBM wrapper
# ---------------------------------------------------------------------------

class LightGBMFraudModel:
    def __init__(self, n_estimators: int = 500, max_depth: int = 7,
                 learning_rate: float = 0.05, class_weight: str = 'balanced',
                 num_leaves: int = 63, random_state: int = 42):
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                class_weight=class_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
            self._available = True
        except ImportError:
            print("lightgbm not installed — using sklearn RandomForest fallback")
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=300, max_depth=8, class_weight='balanced',
                random_state=random_state, n_jobs=-1
            )
            self._available = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "LightGBMFraudModel":
        if self._available and X_val is not None:
            cb = None
            try:
                import lightgbm as lgb
                cb = [lgb.early_stopping(50, verbose=False),
                      lgb.log_evaluation(period=-1)]
            except Exception:
                pass
            self.model.fit(X, y,
                           eval_set=[(X_val, y_val)] if X_val is not None else None,
                           callbacks=cb)
        else:
            self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Stacking ensemble
# ---------------------------------------------------------------------------

class StackingEnsemble:
    """
    Level-0:  XGBoost + LightGBM trained with 5-fold OOF
    Level-1:  Logistic Regression meta-learner on OOF predictions
    
    Out-of-fold (OOF) training prevents leakage from base models to meta-learner.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.xgb_models: list = []
        self.lgb_models: list = []
        self.meta_scaler = StandardScaler()
        self.meta_model  = LogisticRegression(C=1.0, max_iter=1000,
                                               class_weight='balanced',
                                               random_state=random_state)
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple["StackingEnsemble", dict]:
        """
        Trains base models via OOF, then fits meta-learner.
        Returns (self, metrics_dict).
        """
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.random_state)
        N = len(X)
        oof_xgb = np.zeros(N, dtype=np.float32)
        oof_lgb = np.zeros(N, dtype=np.float32)

        print(f"Training stacking ensemble ({self.n_splits}-fold OOF)...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # XGBoost
            xgb_m = XGBoostFraudModel()
            xgb_m.fit(X_tr, y_tr, X_val, y_val)
            oof_xgb[val_idx] = xgb_m.predict_proba(X_val)
            self.xgb_models.append(xgb_m)

            # LightGBM
            lgb_m = LightGBMFraudModel()
            lgb_m.fit(X_tr, y_tr, X_val, y_val)
            oof_lgb[val_idx] = lgb_m.predict_proba(X_val)
            self.lgb_models.append(lgb_m)

            fold_auc = roc_auc_score(y_val, (oof_xgb[val_idx] + oof_lgb[val_idx]) / 2)
            print(f"  Fold {fold+1}/{self.n_splits}  val AUC={fold_auc:.4f}")

        # Train meta-learner on OOF predictions
        meta_X = np.column_stack([oof_xgb, oof_lgb])
        meta_X_scaled = self.meta_scaler.fit_transform(meta_X)
        self.meta_model.fit(meta_X_scaled, y)

        # Evaluate
        meta_probs = self.meta_model.predict_proba(meta_X_scaled)[:, 1]
        metrics = {
            'oof_auc':  float(roc_auc_score(y, meta_probs)),
            'oof_ap':   float(average_precision_score(y, meta_probs)),
            'xgb_coef': float(self.meta_model.coef_[0][0]),
            'lgb_coef': float(self.meta_model.coef_[0][1]),
        }
        print(f"\nStacking OOF AUC={metrics['oof_auc']:.4f}  "
              f"AP={metrics['oof_ap']:.4f}")
        return self, metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns fraud probability from meta-learner."""
        xgb_scores = np.mean([m.predict_proba(X) for m in self.xgb_models], axis=0)
        lgb_scores  = np.mean([m.predict_proba(X) for m in self.lgb_models], axis=0)
        meta_X = np.column_stack([xgb_scores, lgb_scores])
        meta_X_scaled = self.meta_scaler.transform(meta_X)
        return self.meta_model.predict_proba(meta_X_scaled)[:, 1]

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "StackingEnsemble":
        with open(path, 'rb') as f:
            return pickle.load(f)
