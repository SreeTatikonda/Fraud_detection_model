"""
High-fraud scenario test client for the ATO Fraud Detection API.
Seeds realistic account history then fires progressively more suspicious events.

Run:
    Terminal 1:  python -m fraud_detection.api
    Terminal 2:  python -m fraud_detection.test_api
"""

import time
import json
import urllib.request

BASE_URL = "http://localhost:8080"


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def get(path):
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=5) as resp:
        return json.loads(resp.read())


def print_result(label, result):
    action = result["action"]
    score  = result["risk_score"]
    colors = {"ALLOW": "\033[92m", "STEP_UP_MFA": "\033[93m", "BLOCK": "\033[91m"}
    reset  = "\033[0m"
    c      = colors.get(action, "")
    bar    = "#" * int(score * 30) + "." * (30 - int(score * 30))
    print(f"  {label:<45} [{bar}] {score:.3f}  {c}{action}{reset}  ({result['latency_ms']:.1f}ms)")


def seed(account_id, n, now, ip, device, country, city, lat, lon):
    for i in range(n):
        post("/score", {
            "event_id": f"seed_{account_id}_{i}", "account_id": account_id,
            "timestamp": now - (86400 * (i + 1)), "ip_address": ip,
            "device_fingerprint": device, "country": country, "city": city,
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "action": "login", "success": True,
            "latitude": lat, "longitude": lon, "failed_attempts_24h": 0,
            "keystroke_dwell_ms": [115, 122, 108, 131, 119],
            "mouse_velocity_px_s": 480.0, "session_duration_s": 42.0,
        })


def section(title):
    print(f"\n{title}")
    print("-" * 72)


def run():
    print("=" * 72)
    print("  ATO Fraud Detection  —  High Fraud Scenario Test")
    print("=" * 72)

    try:
        h = get("/health")
        print(f"\nAPI: {h['status']}   models_loaded={h['models_loaded']}")
        t = get("/thresholds")
        print(f"Thresholds: {t['routing']}\n")
    except Exception:
        print(f"\nCannot reach API at {BASE_URL}")
        print("Start it with:  python -m fraud_detection.api")
        return

    now = time.time()

    print("Seeding account histories (10 logins each)...")
    seed("acc_clean",      10, now, "72.21.215.90",  "fp_macbook",      "US", "Seattle",      47.6062,   -122.3321)
    seed("acc_victim",     10, now, "98.100.12.55",  "fp_windows_pc",   "US", "Chicago",      41.8781,    -87.6298)
    seed("acc_business",   10, now, "203.45.67.89",  "fp_corp_laptop",  "SG", "Singapore",     1.3521,   103.8198)
    seed("acc_executive",  10, now, "17.253.144.10", "fp_exec_mac",     "US", "New York",     40.7128,    -74.0060)
    seed("acc_travel",     10, now, "80.94.163.80",  "fp_travel_phone", "DE", "Berlin",       52.5200,    13.4050)
    print("Done.\n")

    # ---------------------------------------------------------------
    section("SCENARIO 1  Normal logins (baseline — all should ALLOW)")
    for label, ev in [
        ("Clean US account, known macbook", {
            "event_id": "sc1_001", "account_id": "acc_clean",
            "timestamp": now, "ip_address": "72.21.215.90",
            "device_fingerprint": "fp_macbook", "country": "US", "city": "Seattle",
            "user_agent": "Mozilla/5.0 (Macintosh)", "action": "login", "success": True,
            "latitude": 47.6062, "longitude": -122.3321, "failed_attempts_24h": 0,
            "keystroke_dwell_ms": [112, 118, 105, 128, 116],
            "mouse_velocity_px_s": 460.0, "session_duration_s": 38.0}),
        ("Singapore business account, corp laptop", {
            "event_id": "sc1_002", "account_id": "acc_business",
            "timestamp": now, "ip_address": "203.45.67.89",
            "device_fingerprint": "fp_corp_laptop", "country": "SG", "city": "Singapore",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0)", "action": "login", "success": True,
            "latitude": 1.3521, "longitude": 103.8198, "failed_attempts_24h": 0,
            "keystroke_dwell_ms": [108, 115, 102, 121, 110],
            "mouse_velocity_px_s": 510.0, "session_duration_s": 55.0}),
        ("Frequent traveler, Berlin, known phone", {
            "event_id": "sc1_003", "account_id": "acc_travel",
            "timestamp": now, "ip_address": "80.94.163.80",
            "device_fingerprint": "fp_travel_phone", "country": "DE", "city": "Berlin",
            "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0)",
            "action": "login", "success": True,
            "latitude": 52.5200, "longitude": 13.4050, "failed_attempts_24h": 0,
            "keystroke_dwell_ms": [118, 125, 111, 134, 122],
            "mouse_velocity_px_s": 420.0, "session_duration_s": 32.0}),
    ]:
        print_result(label, post("/score", ev))

    # ---------------------------------------------------------------
    section("SCENARIO 2  Moderate risk (expect ALLOW or STEP_UP_MFA)")
    for label, ev in [
        ("New iPhone, same home city", {
            "event_id": "sc2_001", "account_id": "acc_clean",
            "timestamp": now, "ip_address": "72.21.215.90",
            "device_fingerprint": "fp_new_iphone", "country": "US", "city": "Seattle",
            "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0)",
            "action": "login", "success": True,
            "latitude": 47.6062, "longitude": -122.3321, "failed_attempts_24h": 0,
            "keystroke_dwell_ms": [140, 155, 148, 162, 151],
            "mouse_velocity_px_s": 200.0, "session_duration_s": 18.0}),
        ("3 failed attempts then success", {
            "event_id": "sc2_002", "account_id": "acc_victim",
            "timestamp": now, "ip_address": "98.100.12.55",
            "device_fingerprint": "fp_windows_pc", "country": "US", "city": "Chicago",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0)", "action": "login", "success": True,
            "latitude": 41.8781, "longitude": -87.6298, "failed_attempts_24h": 3,
            "keystroke_dwell_ms": [125, 132, 119, 141, 128],
            "mouse_velocity_px_s": 390.0, "session_duration_s": 30.0}),
        ("Transfer at 3am, known device", {
            "event_id": "sc2_003", "account_id": "acc_clean",
            "timestamp": now - 10800,
            "ip_address": "72.21.215.90", "device_fingerprint": "fp_macbook",
            "country": "US", "city": "Seattle",
            "user_agent": "Mozilla/5.0 (Macintosh)", "action": "transfer", "success": True,
            "latitude": 47.6062, "longitude": -122.3321, "failed_attempts_24h": 0,
            "mouse_velocity_px_s": 320.0, "session_duration_s": 8.0}),
        ("Legit travel: exec in London for conference", {
            "event_id": "sc2_004", "account_id": "acc_executive",
            "timestamp": now, "ip_address": "86.98.10.55",
            "device_fingerprint": "fp_exec_mac", "country": "GB", "city": "London",
            "user_agent": "Mozilla/5.0 (Macintosh)", "action": "login", "success": True,
            "latitude": 51.5074, "longitude": -0.1278, "failed_attempts_24h": 0,
            "keystroke_dwell_ms": [110, 117, 103, 125, 114],
            "mouse_velocity_px_s": 440.0, "session_duration_s": 45.0}),
    ]:
        print_result(label, post("/score", ev))

    # ---------------------------------------------------------------
    section("SCENARIO 3  High risk (expect STEP_UP_MFA or BLOCK)")
    for label, ev in [
        ("Impossible travel: Chicago to Moscow 2h", {
            "event_id": "sc3_001", "account_id": "acc_victim",
            "timestamp": now, "ip_address": "185.220.101.45",
            "device_fingerprint": "fp_unknown_mob", "country": "RU", "city": "Moscow",
            "user_agent": "Mozilla/5.0 (Linux; Android 11)", "action": "login", "success": True,
            "latitude": 55.7558, "longitude": 37.6176, "failed_attempts_24h": 0,
            "keystroke_dwell_ms": [310, 285, 342, 298, 321],
            "mouse_velocity_px_s": 850.0, "session_duration_s": 6.0}),
        ("Scripted agent, 12 failed, Tor exit node", {
            "event_id": "sc3_002", "account_id": "acc_victim",
            "timestamp": now + 60, "ip_address": "185.220.101.45",
            "device_fingerprint": "fp_unknown_002", "country": "RU", "city": "Moscow",
            "user_agent": "python-requests/2.28.0", "action": "login", "success": False,
            "latitude": 55.7558, "longitude": 37.6176, "failed_attempts_24h": 12,
            "keystroke_dwell_ms": [800, 800, 800, 800, 800],
            "mouse_velocity_px_s": 0.0, "session_duration_s": 1.2}),
        ("Headless browser, datacenter IP, pw_reset", {
            "event_id": "sc3_003", "account_id": "acc_clean",
            "timestamp": now, "ip_address": "104.21.34.56",
            "device_fingerprint": "fp_headless_001", "country": "US", "city": "San Francisco",
            "user_agent": "HeadlessChrome/114.0.0.0", "action": "pw_reset", "success": True,
            "latitude": 37.7749, "longitude": -122.4194, "failed_attempts_24h": 5,
            "keystroke_dwell_ms": [500, 500, 500, 500, 500],
            "mouse_velocity_px_s": 0.0, "session_duration_s": 3.5}),
        ("pw_reset after impossible travel", {
            "event_id": "sc3_004", "account_id": "acc_victim",
            "timestamp": now + 120, "ip_address": "185.220.101.45",
            "device_fingerprint": "fp_unknown_mob", "country": "RU", "city": "Moscow",
            "user_agent": "Mozilla/5.0 (Linux; Android 11)", "action": "pw_reset", "success": True,
            "latitude": 55.7558, "longitude": 37.6176, "failed_attempts_24h": 12,
            "keystroke_dwell_ms": [290, 310, 275, 330, 295],
            "mouse_velocity_px_s": 120.0, "session_duration_s": 9.0}),
        ("Credential stuffing: uniform 800ms keystrokes", {
            "event_id": "sc3_005", "account_id": "acc_business",
            "timestamp": now, "ip_address": "198.54.117.200",
            "device_fingerprint": "fp_bot_device", "country": "US", "city": "Dallas",
            "user_agent": "Mozilla/5.0 (compatible; bot)", "action": "login", "success": True,
            "latitude": 32.7767, "longitude": -96.7970, "failed_attempts_24h": 7,
            "keystroke_dwell_ms": [800, 800, 800, 800, 800],
            "mouse_velocity_px_s": 0.0, "session_duration_s": 2.0}),
    ]:
        print_result(label, post("/score", ev))

    # ---------------------------------------------------------------
    section("SCENARIO 4  BLOCK scenarios — all signals firing")
    for label, ev in [
        ("Tor IP + scripted + new device + 15 fails + pw_reset", {
            "event_id": "sc4_001", "account_id": "acc_victim",
            "timestamp": now + 180, "ip_address": "185.220.101.45",
            "device_fingerprint": "fp_block_001", "country": "CN", "city": "Beijing",
            "user_agent": "python-requests/2.28.0", "action": "pw_reset", "success": True,
            "latitude": 39.9042, "longitude": 116.4074, "failed_attempts_24h": 15,
            "keystroke_dwell_ms": [1000, 1000, 1000, 1000, 1000],
            "mouse_velocity_px_s": 0.0, "session_duration_s": 2.1}),
        ("Large transfer, new device, Nigeria, 8 fails", {
            "event_id": "sc4_002", "account_id": "acc_clean",
            "timestamp": now + 300, "ip_address": "198.54.117.200",
            "device_fingerprint": "fp_block_002", "country": "NG", "city": "Lagos",
            "user_agent": "python-requests/2.28.0", "action": "transfer", "success": True,
            "latitude": 6.5244, "longitude": 3.3792, "failed_attempts_24h": 8,
            "keystroke_dwell_ms": [900, 920, 880, 950, 910],
            "mouse_velocity_px_s": 0.0, "session_duration_s": 4.5}),
        ("MFA bypass, headless, 20 fails, Moscow", {
            "event_id": "sc4_003", "account_id": "acc_business",
            "timestamp": now, "ip_address": "104.21.34.56",
            "device_fingerprint": "fp_block_003", "country": "RU", "city": "Moscow",
            "user_agent": "HeadlessChrome/114.0.0.0", "action": "mfa_bypass", "success": False,
            "latitude": 55.7558, "longitude": 37.6176, "failed_attempts_24h": 20,
            "keystroke_dwell_ms": [600, 600, 600, 600, 600],
            "mouse_velocity_px_s": 0.0, "session_duration_s": 1.8}),
        ("Account takeover: exec in North Korea", {
            "event_id": "sc4_004", "account_id": "acc_executive",
            "timestamp": now, "ip_address": "175.45.176.3",
            "device_fingerprint": "fp_block_004", "country": "KP", "city": "Pyongyang",
            "user_agent": "python-requests/2.28.0", "action": "transfer", "success": True,
            "latitude": 39.0392, "longitude": 125.7625, "failed_attempts_24h": 18,
            "keystroke_dwell_ms": [1200, 1200, 1200, 1200, 1200],
            "mouse_velocity_px_s": 0.0, "session_duration_s": 1.5}),
        ("Impossible travel: Berlin to Tehran in 30 min", {
            "event_id": "sc4_005", "account_id": "acc_travel",
            "timestamp": now + 1800, "ip_address": "185.51.201.133",
            "device_fingerprint": "fp_block_005", "country": "IR", "city": "Tehran",
            "user_agent": "python-requests/2.28.0", "action": "pw_reset", "success": True,
            "latitude": 35.6892, "longitude": 51.3890, "failed_attempts_24h": 10,
            "keystroke_dwell_ms": [950, 980, 940, 1010, 970],
            "mouse_velocity_px_s": 0.0, "session_duration_s": 3.2}),
        ("Rapid fire: 30 login attempts in 60 seconds", {
            "event_id": "sc4_006", "account_id": "acc_victim",
            "timestamp": now + 400, "ip_address": "45.142.212.100",
            "device_fingerprint": "fp_block_006", "country": "UA", "city": "Kyiv",
            "user_agent": "Go-http-client/1.1", "action": "login", "success": False,
            "latitude": 50.4501, "longitude": 30.5234, "failed_attempts_24h": 30,
            "keystroke_dwell_ms": [50, 50, 50, 50, 50],
            "mouse_velocity_px_s": 0.0, "session_duration_s": 0.4}),
        ("SIM swap follow-up: new device, pw_reset, transfer", {
            "event_id": "sc4_007", "account_id": "acc_executive",
            "timestamp": now + 500, "ip_address": "104.21.34.56",
            "device_fingerprint": "fp_block_007", "country": "US", "city": "Miami",
            "user_agent": "HeadlessChrome/114.0.0.0", "action": "transfer", "success": True,
            "latitude": 25.7617, "longitude": -80.1918, "failed_attempts_24h": 6,
            "keystroke_dwell_ms": [700, 710, 690, 720, 705],
            "mouse_velocity_px_s": 0.0, "session_duration_s": 5.0}),
        ("Automated card testing: 0 mouse, robotic keystrokes", {
            "event_id": "sc4_008", "account_id": "acc_clean",
            "timestamp": now + 600, "ip_address": "185.220.101.45",
            "device_fingerprint": "fp_block_008", "country": "RO", "city": "Bucharest",
            "user_agent": "curl/7.88.1", "action": "transfer", "success": True,
            "latitude": 44.4268, "longitude": 26.1025, "failed_attempts_24h": 25,
            "keystroke_dwell_ms": [1000, 1000, 1000, 1000, 1000],
            "mouse_velocity_px_s": 0.0, "session_duration_s": 1.0}),
    ]:
        print_result(label, post("/score", ev))

    # ---------------------------------------------------------------
    section("DEBUG  Branch scores for worst event")
    d = post("/debug/score", {
        "event_id": "sc4_debug", "account_id": "acc_victim",
        "timestamp": now + 180, "ip_address": "185.220.101.45",
        "device_fingerprint": "fp_block_001", "country": "CN", "city": "Beijing",
        "user_agent": "python-requests/2.28.0", "action": "pw_reset", "success": True,
        "latitude": 39.9042, "longitude": 116.4074, "failed_attempts_24h": 15,
        "keystroke_dwell_ms": [1000, 1000, 1000, 1000, 1000],
        "mouse_velocity_px_s": 0.0, "session_duration_s": 2.1,
    })
    print(f"  History events:       {d['history_events']}")
    print(f"  Non-zero features:    {d['feature_vector_nonzero']} / 35")
    print(f"  GNN:                  {d['branch_scores']['gnn']}")
    print(f"  Isolation Forest:     {d['branch_scores']['isolation_forest']}")
    print(f"  Autoencoder:          {d['branch_scores']['autoencoder']}")
    print(f"  Combined anomaly:     {d['branch_scores']['anomaly']}")
    print(f"  Ensemble (XGB+LGB):   {d['branch_scores']['ensemble']}")
    print(f"  Final risk score:     {d['final_risk_score']}")
    print(f"  Action:               {d['action']}")

    h2 = get("/health")
    stats = h2.get("latency_stats", {})
    if stats:
        print(f"\nLatency  p50={stats.get('p50_ms')}ms  "
              f"p95={stats.get('p95_ms')}ms  "
              f"p99={stats.get('p99_ms')}ms  "
              f"n={stats.get('count')}")
    print("\nDone.")


if __name__ == "__main__":
    run()