# ATO Fraud Detection System

---

## Business problem

Account takeover fraud occurs when an attacker gains unauthorized access to a legitimate user account, typically through credential theft, brute force, phishing, or session hijacking. Once inside, attackers change passwords, drain balances, initiate fraudulent transfers, or use the account as a relay for further attacks.

Traditional rule-based systems catch known attack patterns but miss novel ones. Static velocity checks generate high false positive rates that damage customer experience. And no single signal — device fingerprint, geolocation, or failed attempt count — is sufficient on its own. Sophisticated attackers rotate IPs, spoof user agents, and pace their attempts to stay below single-signal thresholds.

The cost of inaction is significant. Account takeover fraud costs the financial services industry over 11 billion dollars annually. A missed fraud event typically costs 50 to 100 times more than a false positive MFA challenge. The asymmetry demands a system that is highly sensitive to fraud while keeping the false positive rate low enough that legitimate users are not routinely disrupted.

---

## Deliverables

A trained three-branch machine learning pipeline that assigns a risk score between 0 and 1 to every authentication and transaction event in real time. The pipeline combines a graph neural network operating on the account relationship graph, an unsupervised anomaly detector trained on normal session behavior, and a supervised gradient-boosted ensemble trained on labeled fraud events.

A production-ready REST API that accepts event data and returns a risk score and routing action within 50ms at P95. The API exposes endpoints for single event scoring, batch scoring, threshold management, and per-branch debug output.

A test suite covering four risk tiers from clean baseline logins to maximum-signal fraud scenarios including impossible travel, scripted brute force, headless browser attacks, SIM swap follow-ups, and automated card testing.

A feedback loop design that routes analyst verdicts back into a label store for periodic model retraining, ensuring the system adapts to new attack patterns over time.

Trained models achieving AUC-ROC 0.9732 and 89 percent fraud detection rate on 284,807 real credit card transactions, with 82 of 98 fraud cases in the test set correctly blocked outright.

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
|   score < 0.25    ALLOW                                          |
|   0.25 to 0.60    STEP_UP_MFA                                    |
|   score > 0.60    BLOCK                                          |
+------------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------------+
|                        Feedback loop                             |
|         Label store   retrains on analyst verdicts               |
+------------------------------------------------------------------+
```

---

## Model performance

Trained on the ULB Credit Card Fraud Detection dataset: 284,807 transactions, 492 fraud cases, 0.17 percent fraud rate.

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9732 |
| Average Precision | 0.6506 |
| Fraud detection rate | 89% |
| Fraud blocked outright | 82 of 98 test cases |
| Fraud challenged with MFA | 6 of 98 test cases |
| Fraud missed | 10 of 98 test cases |
| API P95 latency | under 30ms |

The anomaly detector alone achieves AUC-ROC 0.9519 on unseen transactions without any labeled training data, trained exclusively on normal sessions. The supervised ensemble reaches AUC-ROC 0.9786 using 5-fold out-of-fold stacking with XGBoost and LightGBM.

---

## Risk routing

| Score range | Action | Meaning |
|-------------|--------|---------|
| Below 0.25 | ALLOW | Session proceeds without interruption |
| 0.25 to 0.60 | STEP_UP_MFA | User is challenged with a second factor |
| Above 0.60 | BLOCK | Session terminated, account flagged for analyst review |

Thresholds are hot-reloadable without restarting the service via `PUT /thresholds`.

---

## File structure

```
Fraud_Detection/
    creditcard.csv               Training dataset (284,807 transactions)
    fraud_detection/
        features.py              SessionEvent schema, feature extractors, graph edge builder
        data_loader.py           ULB credit card dataset loader and preprocessor
        api.py                   FastAPI service with all endpoints
        test_api.py              Four-scenario test client with debug output
        models/
            gnn_model.py         GraphSAGE (numpy), AccountGraph, PyTorch Geometric stub
            anomaly_model.py     IsolationForest, NumpyAutoencoder, AnomalyDetector
            ensemble_model.py    XGBoost, LightGBM, StackingEnsemble with OOF training
            risk_scorer.py       RiskMetaScorer, RiskDecisionEngine, RiskDecision
        streaming/
            consumer.py          FraudDetectionPipeline, EventStore, GNNScoreCache
        pipeline/
            train.py             Synthetic data training script
            train_real.py        Real data training script
    models/
        saved/
            anomaly.pkl
            ensemble.pkl
            risk_engine.pkl
```

---

## Setup

**Prerequisites**

Python 3.10 or higher. Confirm that `which python` and `pip install` resolve to the same interpreter before proceeding.

**Install dependencies**

```bash
cd ~/Documents/Fraud_Detection
which python
/that/exact/path -m pip install scikit-learn numpy xgboost lightgbm fastapi uvicorn
```

**Train on real data**

```bash
python -m fraud_detection.pipeline.train_real
```

Training completes in approximately 3 minutes on a laptop CPU.

**Start the API**

```bash
python -m fraud_detection.api
```

**Run the test suite**

Open a second terminal.

```bash
python -m fraud_detection.test_api
```

**Interactive API documentation**

```
http://localhost:8080/docs
```

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /score | Score a single event |
| POST | /score/batch | Score up to 100 events in one call |
| POST | /debug/score | Returns all branch scores for a single event |
| GET | /health | Liveness check and latency percentiles |
| GET | /thresholds | Current routing thresholds |
| PUT | /thresholds | Hot-update thresholds without restart |

**Request fields**

| Field | Type | Description |
|-------|------|-------------|
| event_id | string | Unique event identifier |
| account_id | string | Account being authenticated |
| timestamp | float | Unix epoch seconds |
| ip_address | string | IPv4 address of the request |
| device_fingerprint | string | Stable device identifier |
| country / city | string | Geolocation |
| user_agent | string | Browser or client string |
| action | string | login, transfer, pw_reset, mfa_bypass, profile_update |
| success | boolean | Whether the action succeeded |
| latitude / longitude | float | Used for impossible travel detection |
| failed_attempts_24h | integer | Failed attempts in prior 24 hours |
| keystroke_dwell_ms | list of float | Dwell times between keystrokes |
| mouse_velocity_px_s | float | Mouse movement velocity |
| session_duration_s | float | Total session length in seconds |

**Debug response example**

```json
{
  "account_id": "acc_victim",
  "history_events": 10,
  "feature_vector_nonzero": 15,
  "branch_scores": {
    "gnn": 0.1,
    "anomaly": 0.6,
    "isolation_forest": 0.0,
    "autoencoder": 1.0,
    "ensemble": 0.115
  },
  "final_risk_score": 0.803,
  "action": "BLOCK"
}
```

---

## Feature engineering

| Group | Dimensions | Key signals |
|-------|------------|-------------|
| Behavioral | 20 | Keystroke dwell time, mouse velocity, login velocity, time-of-day deviation, failed attempt count |
| Device / network | 15 | Device novelty, impossible travel speed, country change, datacenter IP flag, headless browser detection |
| Graph | per node | Shared device edges, shared IP edges, weighted by connection type |

---

## Production roadmap

Replace EventStore with Redis using ZADD and ZRANGEBYSCORE with a 30-day TTL so history persists across restarts and is shared between workers.

Swap the numpy GraphSAGE with FraudGNNTorch in `models/gnn_model.py` for GPU-backed mini-batch training at production scale.

Replace heuristic datacenter IP detection with MaxMind GeoIP2 or IPInfo for accurate VPN detection, ASN lookup, and precise geolocation.

Containerize the API to eliminate Python environment conflicts across deployments.

```bash
uvicorn fraud_detection.api:app --workers 4 --port 8080
```

Track fraud rate, mean risk score, and action distribution per hour. Alert if BLOCK rate drops to zero for more than one hour.

Route analyst verdicts to a label store and trigger weekly retraining. A 10 percent improvement in label quality typically yields a 3 to 5 point improvement in average precision.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| scikit-learn | 1.3 or higher | IsolationForest, LogisticRegression, StandardScaler |
| numpy | 1.24 or higher | Feature vectors, matrix operations, GNN forward pass |
| xgboost | 1.7 or higher | Gradient-boosted base model |
| lightgbm | 4.0 or higher | Leaf-wise gradient-boosted base model |
| fastapi | 0.100 or higher | Async HTTP API framework |
| uvicorn | 0.22 or higher | ASGI server |
| kafka-python | optional | Kafka consumer and producer for streaming mode |
| torch + torch_geometric | optional | Production GNN with GPU-backed SAGEConv |
