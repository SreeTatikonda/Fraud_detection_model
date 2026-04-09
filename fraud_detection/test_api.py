"""
Test client for the ATO Fraud Detection API.
Sends a mix of legit and suspicious events and prints results.

Run AFTER starting the API:
    Terminal 1:  python -m fraud_detection.api
    Terminal 2:  python -m fraud_detection.test_api
"""

import time
import json
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8080"


def post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=5) as resp:
        return json.loads(resp.read())


def print_decision(label: str, result: dict):
    action = result["action"]
    score  = result["risk_score"]
    color  = {"ALLOW": "\033[92m", "STEP_UP_MFA": "\033[93m", "BLOCK": "\033[91m"}
    reset  = "\033[0m"
    c      = color.get(action, "")
    print(f"  {label:<30} score={score:.3f}  {c}{action}{reset}  "
          f"({result['latency_ms']:.1f}ms)")


def run_tests():
    print("=" * 65)
    print("  ATO Fraud Detection API — Test Client")
    print("=" * 65)

    # --- Health check ---
    try:
        h = get("/health")
        print(f"\nHealth: {h['status']}  models_loaded={h['models_loaded']}")
        if h["latency_stats"]:
            print(f"Latency stats: {h['latency_stats']}")
    except Exception as e:
        print(f"\nCould not reach API at {BASE_URL}. Is it running?")
        print(f"Start it with:  python -m fraud_detection.api")
        return

    # --- Thresholds ---
    t = get("/thresholds")
    print(f"\nRouting thresholds: {t['routing']}\n")

    now = time.time()

    # ----------------------------------------------------------------
    # Test events
    # ----------------------------------------------------------------
    events = [
        # Normal login — same device, same country, daytime
        {
            "label": "Normal login (US, known device)",
            "event_id": "evt_test_001",
            "account_id": "acc_normal_01",
            "timestamp": now,
            "ip_address": "72.21.215.90",
            "device_fingerprint": "fp_known_device_abc",
            "country": "US",
            "city": "Seattle",
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "action": "login",
            "success": True,
            "latitude": 47.6062,
            "longitude": -122.3321,
            "failed_attempts_24h": 0,
        },
        # Suspicious: many failed attempts + new device
        {
            "label": "Brute force attempt",
            "event_id": "evt_test_002",
            "account_id": "acc_victim_01",
            "timestamp": now,
            "ip_address": "185.220.101.45",   # known Tor exit range
            "device_fingerprint": "fp_new_device_xyz",
            "country": "RU",
            "city": "Moscow",
            "user_agent": "python-requests/2.28.0",  # scripted
            "action": "login",
            "success": False,
            "latitude": 55.7558,
            "longitude": 37.6176,
            "failed_attempts_24h": 12,
        },
        # Impossible travel: previous event was US, now Nigeria 2 min later
        {
            "label": "Impossible travel",
            "event_id": "evt_test_003",
            "account_id": "acc_normal_01",   # same account as evt_001
            "timestamp": now + 120,           # only 2 minutes later
            "ip_address": "197.210.55.20",
            "device_fingerprint": "fp_unknown_mobile",
            "country": "NG",
            "city": "Lagos",
            "user_agent": "Mozilla/5.0 (Linux; Android 11)",
            "action": "transfer",
            "success": True,
            "latitude": 6.5244,
            "longitude": 3.3792,
            "failed_attempts_24h": 0,
        },
        # Password reset from datacenter IP
        {
            "label": "pw_reset from datacenter IP",
            "event_id": "evt_test_004",
            "account_id": "acc_victim_02",
            "timestamp": now,
            "ip_address": "104.21.34.56",   # Cloudflare/datacenter range
            "device_fingerprint": "fp_headless_chrome",
            "country": "US",
            "city": "San Francisco",
            "user_agent": "HeadlessChrome/114.0.0.0",
            "action": "pw_reset",
            "success": True,
            "failed_attempts_24h": 3,
        },
        # Normal mobile login
        {
            "label": "Normal mobile login",
            "event_id": "evt_test_005",
            "account_id": "acc_normal_02",
            "timestamp": now,
            "ip_address": "98.100.12.55",
            "device_fingerprint": "fp_iphone_user",
            "country": "US",
            "city": "Chicago",
            "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0)",
            "action": "login",
            "success": True,
            "latitude": 41.8781,
            "longitude": -87.6298,
            "failed_attempts_24h": 0,
        },
    ]

    print("Scoring events:\n")
    for ev in events:
        label = ev.pop("label")
        try:
            result = post("/score", ev)
            print_decision(label, result)
        except Exception as e:
            print(f"  {label:<30} ERROR: {e}")

    # --- Batch endpoint ---
    print("\nBatch scoring (3 events at once):")
    batch = [events[0], events[1], events[4]] if len(events) >= 5 else events[:3]
    try:
        results = post("/score/batch", batch)
        for r in results:
            print(f"  {r['account_id']:<20} score={r['risk_score']:.3f}  {r['action']}")
    except Exception as e:
        print(f"  Batch error: {e}")

    # --- Final health with latency stats ---
    print()
    h2 = get("/health")
    stats = h2.get("latency_stats", {})
    if stats:
        print(f"Latency — p50={stats.get('p50_ms')}ms  "
              f"p95={stats.get('p95_ms')}ms  "
              f"p99={stats.get('p99_ms')}ms")

    print("\nDone.")
    


if __name__ == "__main__":
    run_tests()
