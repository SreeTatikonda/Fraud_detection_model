"""
Real-time streaming inference for ATO fraud detection.
Consumes events from a Kafka topic, runs all three model branches,
and emits risk decisions to an output topic.

Architecture:
  Kafka consumer (async)
    → Feature extraction (sync, <1ms)
    → GNN score lookup (pre-computed node embeddings, <2ms)
    → Anomaly score (in-memory model, <5ms)
    → Ensemble score (in-memory model, <10ms)
    → Risk decision (meta-scorer, <1ms)
    → Kafka producer (async output)

Target P95 latency: <50ms end-to-end (excluding Kafka I/O).

Requirements: kafka-python, asyncio
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict
from typing import Callable, Deque, Dict, List, Optional

from fraud_detection.features import SessionEvent, build_feature_vector
from fraud_detection.models.anomaly_model import AnomalyDetector
from fraud_detection.models.ensemble_model import StackingEnsemble
from fraud_detection.models.gnn_model import FraudGNN, AccountGraph
from fraud_detection.models.risk_scorer import RiskDecisionEngine, RiskDecision

import numpy as np

logger = logging.getLogger("fraud_detector")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# In-memory event store (replaces DB for this demo)
# ---------------------------------------------------------------------------

class EventStore:
    """
    Keeps a rolling 30-day window of events per account.
    In production: Redis with TTL, or a feature store (Feast/Tecton).
    """

    def __init__(self, window_seconds: int = 30 * 86400, max_per_account: int = 500):
        self._store: Dict[str, Deque[SessionEvent]] = defaultdict(
            lambda: deque(maxlen=max_per_account)
        )
        self.window = window_seconds

    def add(self, event: SessionEvent):
        self._store[event.account_id].append(event)

    def get_history(self, account_id: str, since: Optional[float] = None) -> List[SessionEvent]:
        events = list(self._store[account_id])
        if since:
            events = [e for e in events if e.timestamp >= since]
        return events

    def purge_old(self, current_ts: float):
        cutoff = current_ts - self.window
        for acc in list(self._store.keys()):
            while self._store[acc] and self._store[acc][0].timestamp < cutoff:
                self._store[acc].popleft()


# ---------------------------------------------------------------------------
# GNN score cache (node embeddings pre-computed in batch)
# ---------------------------------------------------------------------------

class GNNScoreCache:
    """
    GNN is expensive to run per-event (requires graph rebuild).
    Instead, we run it in batch every N minutes and cache per-account scores.
    Per-event lookup is O(1).
    """

    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[str, float] = {}
        self._last_refresh: float = 0.0
        self.ttl = ttl_seconds

    def get(self, account_id: str, default: float = 0.1) -> float:
        return self._cache.get(account_id, default)

    def refresh(self, event_store: EventStore, gnn: FraudGNN,
                feat_builder: Callable):
        """Rebuild graph and recompute all account scores."""
        logger.info("GNN cache refresh starting...")
        t0 = time.time()

        graph = AccountGraph(feat_dim=35)
        all_events: List[SessionEvent] = []

        # Collect all recent events
        for acc_id, events in event_store._store.items():
            for e in events:
                all_events.append(e)
                history = event_store.get_history(acc_id)
                fv = feat_builder(e, history)
                graph.add_account(acc_id, fv)

        # Build edges (shared device / IP)
        from fraud_detection.features import build_graph_edges, GraphEdge
        edges: List[GraphEdge] = build_graph_edges(all_events)
        for edge in edges:
            graph.add_edge(edge.src, edge.dst, edge.weight)

        if len(graph.account_index) > 0:
            scores = graph.get_scores(gnn)
            self._cache.update(scores)

        self._last_refresh = time.time()
        logger.info(f"GNN cache refreshed {len(self._cache)} accounts "
                    f"in {time.time()-t0:.2f}s")

    def needs_refresh(self) -> bool:
        return (time.time() - self._last_refresh) > self.ttl


# ---------------------------------------------------------------------------
# Core inference pipeline
# ---------------------------------------------------------------------------

class FraudDetectionPipeline:
    """
    Main inference pipeline. Call .process(event) per incoming event.
    Designed for async execution — non-blocking model calls.
    """

    def __init__(self,
                 anomaly_detector: AnomalyDetector,
                 ensemble: StackingEnsemble,
                 gnn: FraudGNN,
                 decision_engine: RiskDecisionEngine,
                 event_store: Optional[EventStore] = None):
        self.anomaly    = anomaly_detector
        self.ensemble   = ensemble
        self.gnn        = gnn
        self.engine     = decision_engine
        self.store      = event_store or EventStore()
        self.gnn_cache  = GNNScoreCache(ttl_seconds=300)

        # Latency tracking
        self._latencies: Deque[float] = deque(maxlen=1000)

    def process(self, event: SessionEvent) -> RiskDecision:
        """Synchronous inference — wraps the full pipeline for one event."""
        t0 = time.perf_counter()

        # 1. Fetch history
        history = self.store.get_history(event.account_id,
                                          since=event.timestamp - 2592000)

        # 2. Build feature vector
        fv = build_feature_vector(event, history)
        fv_2d = fv.reshape(1, -1)

        # 3. Anomaly score (fast — in-memory)
        anomaly_score = float(self.anomaly.score(fv_2d)[0])

        # 4. Supervised ensemble score (fast — in-memory)
        ensemble_score = float(self.ensemble.predict_proba(fv_2d)[0])

        # 5. GNN score (cached — O(1) lookup)
        if self.gnn_cache.needs_refresh():
            self.gnn_cache.refresh(self.store, self.gnn, build_feature_vector)
        gnn_score = self.gnn_cache.get(event.account_id)

        # 6. Risk decision
        decision = self.engine.decide(
            account_id=event.account_id,
            event_id=event.event_id,
            gnn_score=gnn_score,
            anomaly_score=anomaly_score,
            ensemble_score=ensemble_score,
        )

        # 7. Update event store (after scoring to avoid future leakage)
        self.store.add(event)

        total_ms = (time.perf_counter() - t0) * 1000
        self._latencies.append(total_ms)
        decision.latency_ms = round(total_ms, 3)

        logger.info(f"[{event.event_id}] acc={event.account_id} "
                    f"risk={decision.risk_score:.3f} "
                    f"action={decision.action.value} "
                    f"latency={total_ms:.1f}ms")
        return decision

    def get_latency_stats(self) -> dict:
        lats = list(self._latencies)
        if not lats:
            return {}
        return {
            'p50_ms':  round(float(np.percentile(lats, 50)), 2),
            'p95_ms':  round(float(np.percentile(lats, 95)), 2),
            'p99_ms':  round(float(np.percentile(lats, 99)), 2),
            'mean_ms': round(float(np.mean(lats)), 2),
            'count':   len(lats),
        }


# ---------------------------------------------------------------------------
# Kafka streaming consumer (async)
# ---------------------------------------------------------------------------

async def kafka_consumer_loop(pipeline: FraudDetectionPipeline,
                               bootstrap_servers: str = 'localhost:9092',
                               input_topic: str = 'auth-events',
                               output_topic: str = 'fraud-decisions',
                               group_id: str = 'fraud-detector-v1'):
    """
    Async Kafka consumer loop.
    Reads SessionEvent JSON from input_topic, emits RiskDecision JSON to output_topic.

    In production: run multiple instances behind a consumer group for horizontal scaling.
    """
    try:
        from kafka import KafkaConsumer, KafkaProducer
    except ImportError:
        logger.error("kafka-python not installed. Run: pip install kafka-python")
        return

    consumer = KafkaConsumer(
        input_topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset='latest',
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        max_poll_records=50,
    )

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',
        retries=3,
    )

    logger.info(f"Fraud detector listening on {input_topic} → {output_topic}")

    try:
        for msg in consumer:
            try:
                raw = msg.value
                event = SessionEvent(**raw)
                decision = pipeline.process(event)

                # Publish decision
                producer.send(output_topic, value=asdict(decision))

                # Log latency stats every 1000 events
                if len(pipeline._latencies) % 1000 == 0:
                    stats = pipeline.get_latency_stats()
                    logger.info(f"Latency stats: {stats}")

            except Exception as e:
                logger.error(f"Error processing event: {e}", exc_info=True)
                continue

    except KeyboardInterrupt:
        logger.info("Shutting down fraud detector...")
    finally:
        consumer.close()
        producer.close()


# ---------------------------------------------------------------------------
# FastAPI endpoint for synchronous HTTP inference
# ---------------------------------------------------------------------------

FASTAPI_CODE = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="ATO Fraud Detection API")
pipeline: FraudDetectionPipeline = None  # injected at startup

class EventRequest(BaseModel):
    event_id: str
    account_id: str
    timestamp: float
    ip_address: str
    device_fingerprint: str
    country: str
    city: str
    user_agent: str
    action: str
    success: bool

@app.post("/score")
async def score_event(req: EventRequest):
    event = SessionEvent(**req.dict())
    decision = pipeline.process(event)
    return {
        "risk_score":     decision.risk_score,
        "action":         decision.action.value,
        "explanation":    decision.explanation,
        "latency_ms":     decision.latency_ms,
    }

@app.get("/health")
def health():
    return {"status": "ok", "latency_stats": pipeline.get_latency_stats()}

# uvicorn fraud_detection.streaming.consumer:app --workers 4 --port 8080
'''
