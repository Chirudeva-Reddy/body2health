"""
Smoke test script for BodyFit Demo Server
Verifies HTTP endpoints:
- GET /
- GET /api/health
- GET /api/presets
- GET /api/ablation
- GET /api/mesh
- POST /api/predict (Preset & live model forward pass)
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web.server import ThreadedHTTPServer, BodyFitRequestHandler, init_model

TEST_PORT = 8089

def main():
    print("Initializing PyTorch model...")
    init_model()
    
    server = ThreadedHTTPServer(("127.0.0.1", TEST_PORT), BodyFitRequestHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"Test server running on port {TEST_PORT}")
    time.sleep(0.5)

    base_url = f"http://127.0.0.1:{TEST_PORT}"

    try:
        # Test 1: GET /
        print("\n[1/6] Testing GET / (HTML)...")
        with urllib.request.urlopen(f"{base_url}/") as res:
            assert res.status == 200, f"Expected 200, got {res.status}"
            html = res.read().decode("utf-8")
            assert "BodyFit" in html, "HTML missing BodyFit brand title"
            print("  ✓ GET / succeeded (length:", len(html), "bytes)")

        # Test 2: GET /api/health
        print("\n[2/6] Testing GET /api/health...")
        with urllib.request.urlopen(f"{base_url}/api/health") as res:
            assert res.status == 200
            data = json.loads(res.read().decode("utf-8"))
            assert data["status"] == "healthy"
            assert "checkpoint" in data
            print("  ✓ GET /api/health succeeded:", data)

        # Test 3: GET /api/presets
        print("\n[3/6] Testing GET /api/presets...")
        with urllib.request.urlopen(f"{base_url}/api/presets") as res:
            assert res.status == 200
            presets = json.loads(res.read().decode("utf-8"))
            assert "deva_full" in presets
            assert "subject_4577" in presets
            print("  ✓ GET /api/presets succeeded (count:", len(presets), ")")

        # Test 4: GET /api/ablation
        print("\n[4/6] Testing GET /api/ablation...")
        with urllib.request.urlopen(f"{base_url}/api/ablation") as res:
            assert res.status == 200
            ablation = json.loads(res.read().decode("utf-8"))
            assert "limited_ablation" in ablation
            print("  ✓ GET /api/ablation succeeded (rows:", len(ablation["limited_ablation"]), ")")

        # Test 5: GET /api/mesh
        print("\n[5/6] Testing GET /api/mesh...")
        with urllib.request.urlopen(f"{base_url}/api/mesh") as res:
            assert res.status == 200
            content_type = res.headers.get("Content-Type")
            data = res.read()
            assert len(data) > 10000, f"Mesh data too small: {len(data)} bytes"
            print("  ✓ GET /api/mesh succeeded (size:", len(data), "bytes, Content-Type:", content_type, ")")

        # Test 6: POST /api/predict
        print("\n[6/6] Testing POST /api/predict (Live forward pass)...")
        req_body = json.dumps({
            "preset": "deva_full",
            "height_cm": 175.0,
            "sex": "male"
        }).encode("utf-8")
        
        req = urllib.request.Request(
            f"{base_url}/api/predict",
            data=req_body,
            headers={"Content-Type": "application/json"}
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req) as res:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            assert res.status == 200
            pred = json.loads(res.read().decode("utf-8"))
            assert pred["status"] == "success"
            assert "measurements" in pred
            assert "clinical_indices" in pred
            assert "smplx_gate" in pred
            assert "timings" in pred
            print("  ✓ POST /api/predict succeeded in", round(elapsed_ms, 1), "ms")
            print("    Measurements:", pred["measurements"])
            print("    Indices:", pred["clinical_indices"])
            print("    Gate accepted:", pred["smplx_gate"]["accepted"])
            print("    Timings:", pred["timings"])

        print("\n" + "=" * 50)
        print("🎉 ALL DEMO SERVER TESTS PASSED PERFECTLY!")
        print("=" * 50)

    finally:
        server.shutdown()

if __name__ == "__main__":
    main()
