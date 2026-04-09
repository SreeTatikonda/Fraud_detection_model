# Fraud_detection_model


Real-time account takeover detection using a three-branch machine learning pipeline: graph neural network, unsupervised anomaly detection, and a supervised gradient-boosted ensemble. A logistic meta-learner stacks the outputs into a single risk score routed to one of three actions.

---

## Architecture

```
+------------------------------------------------------------------+
|                        Event ingestion                           |
|         Kafka stream   login / device / geolocation / session    |
+------------------------------------------------------------------+
                               |
          +--------------------+--------------------+
          |                    |                    |
+-----------------+  +------------------+  +----------------+
|   Behavioral    |  |  Device/network  |  | Graph features |
|   features      |  |  features        |  |                |
|                 |  |                  |  | Shared device  |
| Keystroke dwell |  | Fingerprint, IP  |  | and IP edges   |
| Nav path, vel.  |  | Travel, VPN flag |  |                |
+-----------------+  +------------------+  +----------------+
          |                    |                    |
          v                    v                    v
+-----------------+  +------------------+  +----------------+
|      GNN        |  | Anomaly detector |  |   Supervised   |
|                 |  |                  |  |   ensemble     |
| GraphSAGE       |  | Isolation Forest |  |                |
| 2-hop neighbor  |  | + Autoencoder    |  | XGBoost +      |
| aggregation     |  | reconstruction   |  | LightGBM OOF   |
+-----------------+  +------------------+  +----------------+
          |                    |                    |
          +--------------------+--------------------+
                               |
                               v
+------------------------------------------------------------------+
|                        Meta-learner                              |
|         Logistic regression on branch scores                     |
|         + interaction terms  gnn x anomaly, ensemble x anomaly  |
+------------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------------+
|                     Risk decision engine                         |
|                                                                  |
|   score < 0.30    ALLOW                                          |
|   0.30 to 0.70    STEP_UP_MFA                                    |
|   score > 0.70    BLOCK                                          |
+------------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------------+
|                        Feedback loop                             |
|         Label store   retrains on analyst verdicts               |
+------------------------------------------------------------------+
```

---

## File structure

```
Fraud_Detection/
    fraud_detection/
        features.py              SessionEvent schema, feature extractors, graph edge builder
        api.py                   FastAPI app with /score, /score/batch, /health, /thresholds, /debug/score
        test_api.py              Test client with history seeding and scenario events
        models/
            gnn_model.py         GraphSAGE (numpy), AccountGraph, PyTorch Geometric stub
            anomaly_model.py     IsolationForest wrapper, NumpyAutoencoder, AnomalyDetector
            ensemble_model.py    XGBoost, LightGBM, StackingEnsemble with OOF training
            risk_scorer.py       RiskMetaScorer, RiskDecisionEngine, RiskDecision dataclass
        streaming/
            consumer.py          FraudDetectionPipeline, EventStore, GNNScoreCache, Kafka loop
        pipeline/
            train.py             End-to-end training script
    models/
        saved/                   Serialized model artifacts
            anomaly.pkl
            ensemble.pkl
            risk_engine.pkl
    venv/
```

---

## Setup

**Prerequisites**

Python 3.10 or higher. All packages must be installed into the same Python interpreter that runs the scripts. Confirm this with `which python` before installing.

**Install dependencies**

```bash
cd ~/Documents/Fraud_Detection
which python
/that/exact/path -m pip install scikit-learn numpy xgboost lightgbm fastapi uvicorn
```

**Train the models**

```bash
cd ~/Documents/Fraud_Detection
python -m fraud_detection.pipeline.train
```

Training on the synthetic dataset completes in under 2 minutes on a laptop CPU. Three pickle files are written to `models/saved/`.

**Start the API**

```bash
python -m fraud_detection.api
```

Expected output:

```
Loading models from disk...
Models loaded. API ready.
Uvicorn running on http://0.0.0.0:8080
```

**Run the test client**

Open a second terminal tab.

```bash
cd ~/Documents/Fraud_Detection
python -m fraud_detection.test_api
```

**Interactive API docs**

With the API running, open the following URL in any browser. FastAPI generates a full Swagger UI with request and response schemas and a live try-it-out interface.

```
http://localhost:8080/docs
```

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /score | Score a single event. Returns risk_score, action, per-model scores, latency. |
| POST | /score/batch | Score up to 100 events in one HTTP call. |
| POST | /debug/score | Returns all intermediate branch scores for a single event. |
| GET | /health | Liveness check and rolling latency percentiles. |
| GET | /thresholds | Returns current ALLOW / STEP_UP_MFA / BLOCK cutoffs. |
| PUT | /thresholds?low=0.3&high=0.7 | Hot-update routing thresholds without restart. |

**Score request fields**

| Field | Type | Notes |
|-------|------|-------|
| event_id | string | Unique identifier for this event |
| account_id | string | Account being authenticated |
| timestamp | float | Unix epoch seconds |
| ip_address | string | IPv4 address of the request |
| device_fingerprint | string | Stable device identifier |
| country / city | string | Geolocation of the request |
| user_agent | string | Browser or client string |
| action | string | login, transfer, pw_reset, mfa_bypass, profile_update |
| success | boolean | Whether the action succeeded |
| latitude / longitude | float, optional | Used for impossible travel detection |
| failed_attempts_24h | integer | Count of failed attempts in prior 24 hours |
| keystroke_dwell_ms | list of float, optional | Dwell times between keystrokes |
| mouse_velocity_px_s | float, optional | Mouse movement velocity |
| session_duration_s | float, optional | Total session length in seconds |

**Debug score response**

```json
{
  "account_id": "acc_normal_01",
  "history_events": 9,
  "feature_vector_nonzero": 15,
  "branch_scores": {
    "gnn": 0.1,
    "anomaly": 0.6,
    "isolation_forest": 0.0,
    "autoencoder": 1.0,
    "ensemble": 0.0002
  },
  "final_risk_score": 0.73,
  "action": "BLOCK"
}
```

---

## Model architecture

**Graph Neural Network**

A 2-layer GraphSAGE network operates on the account-device-IP graph. Node features are 35-dimensional vectors from the feature extractor. Each layer aggregates mean-pooled neighbor embeddings with a learned weight matrix. The output is a per-node fraud probability via sigmoid.

Because graph construction is expensive relative to per-event latency targets, the GNN runs in batch every 5 minutes. Scores are cached in a GNNScoreCache and per-event lookups are O(1).

Production swap: replace `FraudGNN` with `FraudGNNTorch` (PyTorch Geometric SAGEConv) in `models/gnn_model.py`. The numpy version is a drop-in substitute for environments without a GPU.

**Anomaly detection**

Two unsupervised models run on the 35-dimensional feature vector. The Isolation Forest partitions the feature space with random cuts and short average path length signals an outlier. The Autoencoder learns a compressed representation of normal sessions and high reconstruction error on a new session indicates deviation from the learned baseline.

Both are trained exclusively on legitimate sessions. Their scores are combined with weights 0.4 (Isolation Forest) and 0.6 (Autoencoder). The Autoencoder is weighted higher because it captures sequential session structure.

**Supervised ensemble**

XGBoost and LightGBM are trained on labeled fraud events using 5-fold stratified out-of-fold training to prevent leakage into the meta-learner. Both models receive class weight corrections to handle the typical 0.1 to 2 percent fraud rate. A logistic regression meta-learner is trained on the out-of-fold predictions from both base models. Inference averages predictions across all 5 folds for each base model before passing to the meta-learner.

**Risk decision engine**

The RiskMetaScorer combines the three branch scores using a weighted sum with two interaction terms. An escalation boost fires when the anomaly score exceeds 0.5, reflecting that the Autoencoder firing at high confidence is a strong standalone signal. The final score is routed by the RiskDecisionEngine against two configurable thresholds that are hot-reloadable via `PUT /thresholds` without restarting the service.

---

## Feature groups

| Group | Dimensions | Key signals |
|-------|------------|-------------|
| Behavioral | 20 | Keystroke dwell time, mouse velocity, login velocity, time-of-day deviation, failed attempt count |
| Device / network | 15 | Device novelty, impossible travel speed, country change, datacenter IP flag, headless browser detection |
| Graph (edge builder) | per node | Shared device edges, shared IP edges, edge weights by connection type |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| scikit-learn | 1.3 or higher | IsolationForest, LogisticRegression, StandardScaler, metrics |
| numpy | 1.24 or higher | Feature vectors, matrix operations, GNN forward pass |
| xgboost | 1.7 or higher | Gradient-boosted tree base model in the ensemble |
| lightgbm | 4.0 or higher | Leaf-wise gradient-boosted base model in the ensemble |
| fastapi | 0.100 or higher | Async HTTP API framework |
| uvicorn | 0.22 or higher | ASGI server for FastAPI |
| kafka-python | optional | Kafka consumer and producer for streaming mode |
| torch + torch_geometric | optional | Production GNN with GPU-backed SAGEConv |

---

## Production checklist

Replace `generate_synthetic_data` in `train.py` with a loader for your actual labeled dataset. Real transaction data will produce AUC in the 0.92 to 0.97 range depending on feature quality and label noise.

Replace `EventStore` with a Redis client using ZADD and ZRANGEBYSCORE for a TTL-backed shared feature store that persists across restarts and is accessible to multiple API workers.

Replace `FraudGNN` with `FraudGNNTorch` in `models/gnn_model.py` for GPU-backed GraphSAGE with mini-batch neighbor sampling.

Plug in a real IP reputation API such as MaxMind or IPInfo in `extract_device_network_features` in `features.py`. The current datacenter detection uses a placeholder octet check.

Run the API with multiple uvicorn workers behind a load balancer. Each worker maintains its own in-process EventStore so use Redis to share state across them.

```bash
uvicorn fraud_detection.api:app --workers 4 --port 8080
```