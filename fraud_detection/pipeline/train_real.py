"""
End-to-end training pipeline using the real ULB Credit Card Fraud dataset.

Prerequisites:
    Place creditcard.csv in the project root (Fraud_Detection/).

Run:
    python -m fraud_detection.pipeline.train_real
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fraud_detection.data_loader import load_creditcard, load_for_anomaly
from fraud_detection.models.anomaly_model  import AnomalyDetector
from fraud_detection.models.ensemble_model import StackingEnsemble
from fraud_detection.models.gnn_model      import FraudGNN, AccountGraph
from fraud_detection.models.risk_scorer    import RiskMetaScorer, RiskDecisionEngine
from fraud_detection.streaming.consumer    import FraudDetectionPipeline, EventStore


def train():
    print("=" * 60)
    print("ATO Fraud Detection — Real Data Training Pipeline")
    print("=" * 60)

    MODEL_DIR = Path("models/saved")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load real data ----
    print("\n[1/5] Loading creditcard.csv ...")
    X_train, X_test, y_train, y_test = load_creditcard(
        "creditcard.csv", test_size=0.2, random_state=42
    )

    # ---- 2. Train anomaly detector ----
    print("\n[2/5] Training anomaly detector (Isolation Forest + Autoencoder)...")
    print("      Training on legitimate transactions only (semi-supervised)")

    X_legit = X_train[y_train == 0]
    print(f"      Legitimate train samples: {len(X_legit):,}")

    anomaly = AnomalyDetector(if_weight=0.4, ae_weight=0.6)
    anomaly.fit(X_legit)
    anomaly.save(str(MODEL_DIR / "anomaly.pkl"))

    anomaly_scores_test = anomaly.score(X_test)
    anom_auc = roc_auc_score(y_test, anomaly_scores_test)
    anom_ap  = average_precision_score(y_test, anomaly_scores_test)
    print(f"      Anomaly AUC-ROC: {anom_auc:.4f}   AP: {anom_ap:.4f}")

    # ---- 3. Train supervised ensemble ----
    print("\n[3/5] Training supervised ensemble (XGBoost + LightGBM stacking)...")
    print(f"      Class imbalance ratio: {(y_train==0).sum()} legit : {(y_train==1).sum()} fraud")

    ensemble = StackingEnsemble(n_splits=5, random_state=42)
    ensemble, metrics = ensemble.fit(X_train, y_train)
    ensemble.save(str(MODEL_DIR / "ensemble.pkl"))

    ensemble_scores_test = ensemble.predict_proba(X_test)
    ens_auc = roc_auc_score(y_test, ensemble_scores_test)
    ens_ap  = average_precision_score(y_test, ensemble_scores_test)
    print(f"      Ensemble test AUC-ROC: {ens_auc:.4f}   AP: {ens_ap:.4f}")

    # ---- 4. GNN (placeholder on tabular data) ----
    print("\n[4/5] Initializing GNN (graph data not available in tabular dataset)...")
    print("      GNN will use default scores until account graph data is available.")
    print("      In production: feed login events with shared device/IP edges.")
    gnn = FraudGNN(feat_dim=35)
    gnn_scores_train = np.full(len(X_train), 0.1, dtype=np.float32)
    gnn_scores_test  = np.full(len(X_test),  0.1, dtype=np.float32)

    # ---- 5. Train meta-scorer ----
    print("\n[5/5] Training meta-scorer and evaluating final system...")

    anomaly_scores_train  = anomaly.score(X_train)
    ensemble_scores_train = ensemble.predict_proba(X_train)

    meta = RiskMetaScorer(gnn_weight=0.1, anomaly_weight=0.4, ensemble_weight=0.5)
    meta.fit(gnn_scores_train, anomaly_scores_train, ensemble_scores_train, y_train)

    engine = RiskDecisionEngine(
        low_threshold=0.3,
        high_threshold=0.7,
        meta_scorer=meta,
    )

    # Final evaluation
    final_scores = np.array([
        meta.score(float(gnn_scores_test[i]),
                   float(anomaly_scores_test[i]),
                   float(ensemble_scores_test[i]))
        for i in range(len(X_test))
    ])

    final_preds = (final_scores >= 0.5).astype(int)

    print(f"\n{'='*60}")
    print("FINAL SYSTEM PERFORMANCE ON REAL DATA")
    print(f"{'='*60}")
    print(f"  AUC-ROC:           {roc_auc_score(y_test, final_scores):.4f}")
    print(f"  Average Precision: {average_precision_score(y_test, final_scores):.4f}")
    print()
    print(classification_report(y_test, final_preds,
                                  target_names=["Legit", "Fraud"]))

    # Threshold analysis
    print("Routing distribution on test set:")
    allow      = (final_scores < 0.3).sum()
    step_up    = ((final_scores >= 0.3) & (final_scores < 0.7)).sum()
    block      = (final_scores >= 0.7).sum()
    total      = len(final_scores)
    print(f"  ALLOW:       {allow:6,} ({100*allow/total:.1f}%)")
    print(f"  STEP_UP_MFA: {step_up:6,} ({100*step_up/total:.1f}%)")
    print(f"  BLOCK:       {block:6,} ({100*block/total:.1f}%)")

    # Fraud caught breakdown
    fraud_idx = y_test == 1
    fraud_scores = final_scores[fraud_idx]
    print(f"\nOf {fraud_idx.sum()} fraud cases in test set:")
    print(f"  Caught (BLOCK):       {(fraud_scores >= 0.7).sum()}")
    print(f"  Challenged (STEP_UP): {((fraud_scores >= 0.3) & (fraud_scores < 0.7)).sum()}")
    print(f"  Missed (ALLOW):       {(fraud_scores < 0.3).sum()}")

    engine.save(str(MODEL_DIR / "risk_engine.pkl"))
    print(f"\nModels saved to {MODEL_DIR}/")
    return engine


if __name__ == "__main__":
    train()