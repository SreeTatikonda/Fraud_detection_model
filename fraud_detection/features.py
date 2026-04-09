"""
Feature engineering for Account Takeover (ATO) detection.
Produces behavioral, device/network, and graph features from raw session events.
"""

from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Raw event schema
# ---------------------------------------------------------------------------

@dataclass
class SessionEvent:
    """Represents one login/action event from the Kafka stream."""
    event_id: str
    account_id: str
    timestamp: float               # Unix epoch seconds
    ip_address: str
    device_fingerprint: str
    country: str
    city: str
    user_agent: str
    action: str                    # 'login', 'pw_reset', 'transfer', etc.
    success: bool
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    keystroke_dwell_ms: Optional[List[float]] = None   # dwell times between keys
    mouse_velocity_px_s: Optional[float] = None
    session_duration_s: Optional[float] = None
    failed_attempts_24h: int = 0


# ---------------------------------------------------------------------------
# Behavioral feature extractor
# ---------------------------------------------------------------------------

def extract_behavioral_features(event: SessionEvent,
                                 history: List[SessionEvent]) -> np.ndarray:
    """
    Returns a 20-dim behavioral feature vector.
    Captures typing rhythm, session timing, and historical anomalies.
    """
    feats = []

    # --- Keystroke dynamics ---
    if event.keystroke_dwell_ms and len(event.keystroke_dwell_ms) > 1:
        dwell = np.array(event.keystroke_dwell_ms)
        feats += [dwell.mean(), dwell.std(), dwell.max(), dwell.min()]
    else:
        feats += [0.0, 0.0, 0.0, 0.0]

    # --- Mouse / interaction ---
    feats.append(event.mouse_velocity_px_s or 0.0)
    feats.append(event.session_duration_s or 0.0)

    # --- Failed attempts ---
    feats.append(float(event.failed_attempts_24h))
    feats.append(float(event.failed_attempts_24h > 5))  # spike indicator

    # --- Historical velocity (last 24 h) ---
    cutoff = event.timestamp - 86400
    recent = [e for e in history if e.timestamp >= cutoff]
    feats.append(float(len(recent)))                       # login count
    feats.append(float(sum(1 for e in recent if not e.success)))  # fail count

    # --- Time-of-day anomaly (vs user's historical mean hour) ---
    if history:
        hist_hours = [(e.timestamp % 86400) / 3600 for e in history]
        mean_hour = np.mean(hist_hours)
        cur_hour = (event.timestamp % 86400) / 3600
        hour_delta = abs(cur_hour - mean_hour)
        # Wrap around midnight
        hour_delta = min(hour_delta, 24 - hour_delta)
        feats.append(hour_delta)
    else:
        feats.append(0.0)

    # --- Unique IPs / devices in last 7 days ---
    week_ago = event.timestamp - 604800
    week_events = [e for e in history if e.timestamp >= week_ago]
    feats.append(float(len(set(e.ip_address for e in week_events))))
    feats.append(float(len(set(e.device_fingerprint for e in week_events))))

    # --- Action risk weight ---
    action_risk = {'login': 0.1, 'pw_reset': 0.8, 'transfer': 0.9,
                   'mfa_bypass': 1.0, 'profile_update': 0.5}
    feats.append(action_risk.get(event.action, 0.3))

    # --- Day of week (0=Mon, 6=Sun, weekends slightly riskier in ATO) ---
    dow = (int(event.timestamp / 86400) + 4) % 7  # rough day-of-week
    feats.append(float(dow >= 5))

    # Padding to exactly 20 dims
    while len(feats) < 20:
        feats.append(0.0)

    return np.array(feats[:20], dtype=np.float32)


# ---------------------------------------------------------------------------
# Device / network feature extractor
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def extract_device_network_features(event: SessionEvent,
                                     history: List[SessionEvent]) -> np.ndarray:
    """
    Returns a 15-dim device and network feature vector.
    Covers impossible travel, VPN/datacenter IPs, new device signals.
    """
    feats = []

    # --- Device novelty ---
    known_fps = {e.device_fingerprint for e in history}
    feats.append(float(event.device_fingerprint not in known_fps))
    feats.append(float(len(known_fps)))

    # --- Impossible travel ---
    if history and event.latitude and event.longitude:
        last = sorted(history, key=lambda e: e.timestamp)[-1]
        if last.latitude and last.longitude:
            dist_km = haversine_km(last.latitude, last.longitude,
                                   event.latitude, event.longitude)
            dt_h = max((event.timestamp - last.timestamp) / 3600, 1e-3)
            speed = dist_km / dt_h  # km/h
            feats.append(dist_km)
            feats.append(speed)
            feats.append(float(speed > 900))  # faster than commercial flight
        else:
            feats += [0.0, 0.0, 0.0]
    else:
        feats += [0.0, 0.0, 0.0]

    # --- Country change ---
    if history:
        last_country = sorted(history, key=lambda e: e.timestamp)[-1].country
        feats.append(float(last_country != event.country))
    else:
        feats.append(0.0)

    # --- IP reputation proxies (in prod: call MaxMind/IPInfo API) ---
    # Simplified heuristics for demo
    ip_parts = event.ip_address.split('.')
    feats.append(float(len(ip_parts) != 4))      # malformed IP
    # Known datacenter ranges (placeholder — use real IP intel in prod)
    datacenter_octets = {'104', '198', '185'}
    feats.append(float(ip_parts[0] in datacenter_octets))

    # --- User-agent anomaly ---
    ua = event.user_agent.lower()
    feats.append(float('bot' in ua or 'crawler' in ua or 'python' in ua))
    feats.append(float('headless' in ua))

    # --- IP consistency in last 30 days ---
    month_ago = event.timestamp - 2592000
    month_ips = {e.ip_address for e in history if e.timestamp >= month_ago}
    feats.append(float(event.ip_address not in month_ips))
    feats.append(float(len(month_ips)))

    # Padding
    while len(feats) < 15:
        feats.append(0.0)

    return np.array(feats[:15], dtype=np.float32)


# ---------------------------------------------------------------------------
# Combined feature vector
# ---------------------------------------------------------------------------

def build_feature_vector(event: SessionEvent,
                          history: List[SessionEvent]) -> np.ndarray:
    """
    Concatenates behavioral (20) + device/network (15) = 35-dim feature vector.
    This is the input to Isolation Forest, Autoencoder, and XGBoost.
    """
    beh = extract_behavioral_features(event, history)
    dev = extract_device_network_features(event, history)
    return np.concatenate([beh, dev])


# ---------------------------------------------------------------------------
# Graph edge builder (for GNN ingestion)
# ---------------------------------------------------------------------------

@dataclass
class GraphEdge:
    src: str        # account_id
    dst: str        # account_id
    edge_type: str  # 'shared_device', 'shared_ip', 'shared_email'
    weight: float = 1.0


def build_graph_edges(events: List[SessionEvent]) -> List[GraphEdge]:
    """
    Builds account-to-account edges based on shared device/IP signals.
    Output is consumed by the GNN module.
    """
    from collections import defaultdict

    device_to_accounts: dict[str, set] = defaultdict(set)
    ip_to_accounts: dict[str, set] = defaultdict(set)

    for e in events:
        device_to_accounts[e.device_fingerprint].add(e.account_id)
        ip_to_accounts[e.ip_address].add(e.account_id)

    edges: List[GraphEdge] = []

    for fp, accounts in device_to_accounts.items():
        accs = list(accounts)
        for i in range(len(accs)):
            for j in range(i + 1, len(accs)):
                edges.append(GraphEdge(accs[i], accs[j], 'shared_device', 1.0))

    for ip, accounts in ip_to_accounts.items():
        accs = list(accounts)
        for i in range(len(accs)):
            for j in range(i + 1, len(accs)):
                edges.append(GraphEdge(accs[i], accs[j], 'shared_ip', 0.7))

    return edges
