"""
FastAPI endpoint for real-time ATO fraud scoring.

Run:
    pip install fastapi uvicorn
    python -m fraud_detection.api

Then test:
    python -m fraud_detection.test_api
"""

from __future__ import annotations
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from fraud_detection.features import SessionEvent, build_feature_vector
from fraud_detection.models.anomaly_model import AnomalyDetector
from fraud_detection.models.ensemble_model import StackingEnsemble
from fraud_detection.models.gnn_model import FraudGNN, AccountGraph
from fraud_detection.models.risk_scorer import RiskDecisionEngine
from fraud_detection.streaming.consumer import FraudDetectionPipeline, EventStore


# ---------------------------------------------------------------------------
# Model registry — loaded once at startup
# ---------------------------------------------------------------------------

MODEL_DIR = Path("models/saved")

registry: dict = {}


def load_models():
    print("Loading models from disk...")

    with open(MODEL_DIR / "anomaly.pkl", "rb") as f:
        registry["anomaly"] = pickle.load(f)

    with open(MODEL_DIR / "ensemble.pkl", "rb") as f:
        registry["ensemble"] = pickle.load(f)

    with open(MODEL_DIR / "risk_engine.pkl", "rb") as f:
        registry["engine"] = pickle.load(f)

    registry["gnn"]      = FraudGNN(feat_dim=35)
    registry["store"]    = EventStore()
    registry["pipeline"] = FraudDetectionPipeline(
        anomaly_detector=registry["anomaly"],
        ensemble=registry["ensemble"],
        gnn=registry["gnn"],
        decision_engine=registry["engine"],
        event_store=registry["store"],
    )
    print("Models loaded. API ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    print("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ATO Fraud Detection API",
    description="Real-time account takeover scoring using GNN + Anomaly + Ensemble stack",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    event_id: str               = Field(..., example="evt_001")
    account_id: str             = Field(..., example="acc_00042")
    timestamp: float            = Field(..., example=1712620800.0)
    ip_address: str             = Field(..., example="192.168.1.1")
    device_fingerprint: str     = Field(..., example="fp_abc123")
    country: str                = Field(..., example="US")
    city: str                   = Field(..., example="New York")
    user_agent: str             = Field(..., example="Mozilla/5.0")
    action: str                 = Field(..., example="login")
    success: bool               = Field(..., example=True)
    latitude: Optional[float]   = Field(None, example=40.7128)
    longitude: Optional[float]  = Field(None, example=-74.0060)
    failed_attempts_24h: int    = Field(0, example=0)


class ScoreResponse(BaseModel):
    event_id: str
    account_id: str
    risk_score: float
    action: str
    gnn_score: float
    anomaly_score: float
    ensemble_score: float
    latency_ms: float
    explanation: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/score", response_model=ScoreResponse)
def score_event(req: ScoreRequest):
    """
    Score a single login/action event for ATO fraud risk.
    Returns a risk score 0–1 and a routing action.
    """
    if "pipeline" not in registry:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    event = SessionEvent(
        event_id=req.event_id,
        account_id=req.account_id,
        timestamp=req.timestamp,
        ip_address=req.ip_address,
        device_fingerprint=req.device_fingerprint,
        country=req.country,
        city=req.city,
        user_agent=req.user_agent,
        action=req.action,
        success=req.success,
        latitude=req.latitude,
        longitude=req.longitude,
        failed_attempts_24h=req.failed_attempts_24h,
    )

    decision = registry["pipeline"].process(event)

    return ScoreResponse(
        event_id=decision.event_id,
        account_id=decision.account_id,
        risk_score=decision.risk_score,
        action=decision.action.value,
        gnn_score=decision.gnn_score,
        anomaly_score=decision.anomaly_score,
        ensemble_score=decision.ensemble_score,
        latency_ms=decision.latency_ms,
        explanation=decision.explanation,
    )


@app.post("/score/batch")
def score_batch(events: list[ScoreRequest]):
    """Score up to 100 events in one call."""
    if len(events) > 100:
        raise HTTPException(status_code=400, detail="Max 100 events per batch")
    return [score_event(e) for e in events]


@app.get("/health")
def health():
    """Liveness + latency stats."""
    pipeline = registry.get("pipeline")
    return {
        "status": "ok",
        "models_loaded": "pipeline" in registry,
        "latency_stats": pipeline.get_latency_stats() if pipeline else {},
    }


@app.get("/thresholds")
def get_thresholds():
    """Return current risk routing thresholds."""
    engine: RiskDecisionEngine = registry.get("engine")
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    return {
        "low_threshold":  engine.low_threshold,
        "high_threshold": engine.high_threshold,
        "routing": {
            f"< {engine.low_threshold}":  "ALLOW",
            f"{engine.low_threshold} – {engine.high_threshold}": "STEP_UP_MFA",
            f"> {engine.high_threshold}": "BLOCK",
        }
    }


@app.put("/thresholds")
def update_thresholds(low: float, high: float):
    """Hot-update risk thresholds without restarting."""
    engine: RiskDecisionEngine = registry.get("engine")
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    try:
        engine.update_thresholds(low, high)
        return {"status": "updated", "low": low, "high": high}
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    @app.post("/debug/score")
def debug_score(req: ScoreRequest):
    if "pipeline" not in registry:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    from fraud_detection.features import build_feature_vector

    pipeline = registry["pipeline"]
    event = SessionEvent(
        event_id=req.event_id, account_id=req.account_id,
        timestamp=req.timestamp, ip_address=req.ip_address,
        device_fingerprint=req.device_fingerprint,
        country=req.country, city=req.city,
        user_agent=req.user_agent, action=req.action,
        success=req.success, latitude=req.latitude,
        longitude=req.longitude, failed_attempts_24h=req.failed_attempts_24h,
    )

    history = pipeline.store.get_history(event.account_id,
                                          since=event.timestamp - 2592000)
    fv = build_feature_vector(event, history)

    anomaly_score  = float(pipeline.anomaly.score(fv.reshape(1, -1))[0])
    ensemble_score = float(pipeline.ensemble.predict_proba(fv.reshape(1, -1))[0])
    gnn_score      = pipeline.gnn_cache.get(event.account_id)
    iso_score      = float(pipeline.anomaly.iso_forest.score(fv.reshape(1, -1))[0])
    ae_score       = float(pipeline.anomaly.autoencoder.score(fv.reshape(1, -1))[0])
    final_score    = pipeline.engine.meta_scorer.score(gnn_score, anomaly_score, ensemble_score)

    return {
        "account_id": event.account_id,
        "history_events": len(history),
        "feature_vector_nonzero": int((fv != 0).sum()),
        "branch_scores": {
            "gnn":              round(gnn_score, 4),
            "anomaly":          round(anomaly_score, 4),
            "isolation_forest": round(iso_score, 4),
            "autoencoder":      round(ae_score, 4),
            "ensemble":         round(ensemble_score, 4),
        },
        "final_risk_score": round(final_score, 4),
        "action": pipeline.engine.decide(
            event.account_id, event.event_id,
            gnn_score, anomaly_score, ensemble_score
        ).action.value,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fraud_detection.api:app", host="0.0.0.0", port=8080, reload=False)