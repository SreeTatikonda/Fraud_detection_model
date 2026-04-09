"""
End-to-end training pipeline for ATO fraud detection.
Generates synthetic data, trains all three branches, evaluates, and saves models.

Run:  python -m fraud_detection.pipeline.train
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fraud_detection.features import (
    SessionEvent, build_feature_vector, build_graph_edges
)
from fraud_detection.models.anomaly_model  import AnomalyDetector
from fraud_detection.models.ensemble_model import StackingEnsemble
from fraud_detection.models.gnn_model      import FraudGNN, AccountGraph
from fraud_detection.models.risk_scorer    import RiskMetaScorer, RiskDecisionEngine
from fraud_detection.streaming.consumer    import FraudDetectionPipeline, EventStore


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_legit: int = 5000, n_fraud: int = 150,
                              seed: int = 42) -> tuple:
    """
    Generates synthetic feature vectors and labels.
    Fraud accounts have higher anomaly signals in selected features.
    Returns (X, y, account_ids).
    """
    rng = np.random.default_rng(seed)

    # Legitimate users
    X_legit = rng.normal(0, 1, (n_legit, 35)).astype(np.float32)
    # Clip to reasonable range
    X_legit = np.clip(X_legit, -3, 3)

    # Fraudsters: elevated features at indices 0,1,7,10,13 (key ATO signals)
    X_fraud = rng.normal(0, 1, (n_fraud, 35)).astype(np.float32)
    fraud_signal_dims = [0, 1, 7, 10, 13, 20, 21, 24]
    for d in fraud_signal_dims:
        X_fraud[:, d] = rng.normal(3.5, 1.0, n_fraud)   # shifted distribution

    X = np.vstack([X_legit, X_fraud])
    y = np.array([0] * n_legit + [1] * n_fraud, dtype=np.int32)
    account_ids = [f"acc_{i:05d}" for i in range(len(y))]

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx], [account_ids[i] for i in idx]


def generate_synthetic_events(n: int = 200, fraud_rate: float = 0.05,
                                seed: int = 42) -> list:
    """Creates minimal SessionEvent objects for GNN graph construction."""
    rng = np.random.default_rng(seed)
    events = []
    devices = [f"dev_{i}" for i in range(20)]
    ips     = [f"10.0.{rng.integers(0, 255)}.{rng.integers(1, 254)}" for _ in range(30)]

    for i in range(n):
        is_fraud = rng.random() < fraud_rate
        # Fraudsters share devices/IPs more
        device_pool = devices[:3] if is_fraud else devices
        ip_pool     = ips[:5]    if is_fraud else ips

        events.append(SessionEvent(
            event_id=f"evt_{i:06d}",
            account_id=f"acc_{rng.integers(0, 80 if is_fraud else 200):05d}",
            timestamp=float(int(time.time()) - rng.integers(0, 86400 * 7)),
            ip_address=rng.choice(ip_pool),
            device_fingerprint=rng.choice(device_pool),
            country='US',
            city='New York',
            user_agent='Mozilla/5.0',
            action=rng.choice(['login', 'transfer', 'pw_reset']),
            success=bool(rng.random() > 0.1),
            latitude=float(rng.uniform(25, 48)),
            longitude=float(rng.uniform(-122, -70)),
        ))
    return events


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train():
    print("=" * 60)
    print("ATO Fraud Detection — Training Pipeline")
    print("=" * 60)

    MODEL_DIR = Path("models/saved")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Generate data ----
    print("\n[1/5] Generating synthetic training data...")
    X, y, account_ids = generate_synthetic_data(n_legit=5000, n_fraud=150)
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    print(f"  Train: {len(X_train)} samples ({y_train.sum()} fraud)")
    print(f"  Test:  {len(X_test)} samples ({y_test.sum()} fraud)")

    # ---- 2. Train anomaly detector ----
    print("\n[2/5] Training anomaly detector (Isolation Forest + Autoencoder)...")
    # Train only on legitimate samples (semi-supervised)
    X_legit = X_train[y_train == 0]
    anomaly = AnomalyDetector()
    anomaly.fit(X_legit)
    anomaly.save(str(MODEL_DIR / "anomaly.pkl"))

    anomaly_scores_test = anomaly.score(X_test)
    print(f"  Anomaly AUC: {roc_auc_score(y_test, anomaly_scores_test):.4f}")

    # ---- 3. Train supervised ensemble ----
    print("\n[3/5] Training supervised ensemble (XGBoost + LightGBM stacking)...")
    ensemble = StackingEnsemble(n_splits=5)
    ensemble, metrics = ensemble.fit(X_train, y_train)
    ensemble.save(str(MODEL_DIR / "ensemble.pkl"))

    ensemble_scores_test = ensemble.predict_proba(X_test)
    print(f"  Ensemble test AUC: {roc_auc_score(y_test, ensemble_scores_test):.4f}")
    print(f"  Ensemble test AP:  {average_precision_score(y_test, ensemble_scores_test):.4f}")

    # ---- 4. Build GNN graph and get scores ----
    print("\n[4/5] Building account graph and computing GNN scores...")
    events = generate_synthetic_events(n=500)
    graph  = AccountGraph(feat_dim=35)
    rng    = np.random.default_rng(0)

    for acc_id in account_ids[:200]:
        fv = rng.normal(0, 1, 35).astype(np.float32)
        graph.add_account(acc_id, fv)

    edges = build_graph_edges(events)
    for edge in edges:
        graph.add_edge(edge.src, edge.dst, edge.weight)

    gnn = FraudGNN(feat_dim=35)
    X_gnn, A_gnn = graph.to_matrices()

    if len(X_gnn) > 0:
        # Create synthetic labels for GNN training
        gnn_labels = np.zeros(len(X_gnn))
        gnn_labels[:10] = 1   # simulate 10 known fraud accounts
        gnn.fit(X_gnn, A_gnn, gnn_labels, lr=0.01, epochs=50)

    gnn_score_map = graph.get_scores(gnn)
    # Map to test set (use 0.1 as default for unknown accounts)
    gnn_scores_test = np.array([
        gnn_score_map.get(account_ids[n_train + i], 0.1)
        for i in range(len(X_test))
    ])

    # ---- 5. Train meta-scorer and evaluate ----
    print("\n[5/5] Training meta-scorer and evaluating final system...")

    # Get scores for all training samples
    anomaly_scores_train  = anomaly.score(X_train)
    ensemble_scores_train = ensemble.predict_proba(X_train)
    gnn_scores_train = np.array([
        gnn_score_map.get(account_ids[i], 0.1)
        for i in range(n_train)
    ])

    meta = RiskMetaScorer()
    meta.fit(gnn_scores_train, anomaly_scores_train, ensemble_scores_train, y_train)

    engine = RiskDecisionEngine(
        low_threshold=0.3,
        high_threshold=0.7,
        meta_scorer=meta,
    )

    # Evaluate on test set
    final_scores = np.array([
        meta.score(float(gnn_scores_test[i]),
                   float(anomaly_scores_test[i]),
                   float(ensemble_scores_test[i]))
        for i in range(len(X_test))
    ])

    final_preds = (final_scores >= 0.5).astype(int)

    print(f"\n{'='*60}")
    print("FINAL SYSTEM PERFORMANCE")
    print(f"{'='*60}")
    print(f"  AUC-ROC:           {roc_auc_score(y_test, final_scores):.4f}")
    print(f"  Average Precision: {average_precision_score(y_test, final_scores):.4f}")
    print()
    print(classification_report(y_test, final_preds, target_names=['Legit', 'Fraud']))

    # Save final engine
    engine.save(str(MODEL_DIR / "risk_engine.pkl"))
    print(f"\nModels saved to {MODEL_DIR}/")

    # ---- Demo: run a few events through the pipeline ----
    print("\n" + "="*60)
    print("DEMO: Real-time scoring (5 events)")
    print("="*60)

    store = EventStore()
    pipeline = FraudDetectionPipeline(
        anomaly_detector=anomaly,
        ensemble=ensemble,
        gnn=gnn,
        decision_engine=engine,
        event_store=store,
    )

    demo_events = generate_synthetic_events(n=5, fraud_rate=0.4, seed=99)
    for evt in demo_events:
        decision = pipeline.process(evt)
        print(f"  {decision.explanation}")

    print("\nDone.")
    return pipeline


if __name__ == "__main__":
    train()
