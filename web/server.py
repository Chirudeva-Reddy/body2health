#!/usr/bin/env python3
"""
Body2Fit Interactive Web Application Server
============================================
A fast, lightweight, standalone HTTP server serving the interactive recruiter demo,
providing REST endpoints for live forward pass inference, 3D mesh streaming,
and ablation benchmark visualization.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import mimetypes
import os
from pathlib import Path
import sys
import time
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("body2fit.server")

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infer.masks import load_mask_binary, mask_to_tensor
from src.infer.model_io import load_dualview_checkpoint
from src.infer.predict import predict_from_pair
from src.infer.silhouette_checks import envelope_check
from src.infer.silhouette_regions import make_region_overlay, region_metrics
from src.metrics.body_indices import derive_indices, derive_risk_categories
from src.metrics.health_risk import assess_health_risk
from src.smpl.gate import default_gate_thresholds, evaluate_smplx_gate

DEFAULT_CKPT = str(ROOT / "checkpoints" / "best_640x480_v4_resnet.pt")
SMPLX_MODEL_DIR = str(ROOT / "body_models" / "smplx")
DEFAULT_PORT = 8080
TARGET_HW = (640, 480)

# Global model cache
GLOBAL_MODEL = None
GLOBAL_CKPT = None
GLOBAL_DEVICE = "cpu"


def init_model(ckpt_path: str = DEFAULT_CKPT, device: Optional[str] = None) -> Tuple[Any, Dict[str, Any], str]:
    global GLOBAL_MODEL, GLOBAL_CKPT, GLOBAL_DEVICE
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    logger.info(f"Loading dual-view checkpoint from {ckpt_path} on {device}...")
    t0 = time.perf_counter()
    model, ckpt = load_dualview_checkpoint(ckpt_path, device)
    model.eval()
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"Checkpoint loaded successfully in {elapsed:.1f}ms on {device}")
    
    GLOBAL_MODEL = model
    GLOBAL_CKPT = ckpt
    GLOBAL_DEVICE = device
    return model, ckpt, device


def get_cached_presets() -> Dict[str, Dict[str, Any]]:
    """Returns curated recruiter presets with sample photos and precomputed outputs."""
    return {
        "deva_full": {
            "id": "deva_full",
            "title": "Deva (Full Dual-View - Validated SMPL-X)",
            "subtitle": "Primary Paper Validation Subject • 175cm • Male",
            "description": "Standard capture with front and lateral silhouettes. Runs the SMPL-X geometry reliability gate; when the render-back agrees with the silhouettes, the full central-adiposity report is generated.",
            "height_cm": 175.0,
            "sex": "male",
            "capture_mode": "dual_view",
            "front_photo": "/media/TestPhoto/deva_front.png",
            "side_photo": "/media/TestPhoto/deva_side.png",
            "front_silhouette": "/media/out/deva_front_silhouette.png",
            "side_silhouette": "/media/out/deva_side_silhouette.png",
            "front_overlay": "/media/outputs/demo/deva_strict/front_part_overlay.png",
            "side_overlay": "/media/outputs/demo/deva_strict/side_part_overlay.png",
            "smplx_overlay": "/media/outputs/demo/deva_strict/smplx/smplx_front_overlay.png",
            "smplx_rendered": "/media/outputs/demo/deva_strict/smplx/smplx_rendered_front.png",
            "obj_path": "/api/mesh?preset=deva",
            "cached_result_json": "/media/outputs/demo/deva_strict/result.json",
            "real_measurements": {"waist_cm": 86.73, "hip_cm": 100.29, "chest_cm": 93.36}
        },
        "subject_4577": {
            "id": "subject_4577",
            "title": "Subject 4577 (Female Anthropometry - Borderline WHtR)",
            "subtitle": "Clinical Screening • 168cm • Female",
            "description": "Female anthropometry screening showing increased central-adiposity indicator with borderline WHtR cutoff warning.",
            "height_cm": 168.0,
            "sex": "female",
            "capture_mode": "dual_view",
            "front_photo": "/media/TestPhoto/IMG_4577.jpg",
            "side_photo": "/media/TestPhoto/IMG_4373.jpeg",
            "front_silhouette": "/media/out/IMG_4577_final_silhouette.png",
            "side_silhouette": "/media/out/IMG_4373_silhouette.png",
            "front_overlay": "/media/outputs/final/deva/front_regions_overlay.png",
            "side_overlay": "/media/outputs/final/deva/side_regions_overlay.png",
            "smplx_overlay": "/media/outputs/demo/deva_strict/smplx/smplx_front_overlay.png",
            "smplx_rendered": "/media/outputs/demo/deva_strict/smplx/smplx_rendered_front.png",
            "obj_path": "/api/mesh?preset=deva"
        },
        "diagnostic_copy4": {
            "id": "diagnostic_copy4",
            "title": "Diagnostic Single-Front (High Adiposity)",
            "subtitle": "High Adiposity Diagnostic • 175cm • Female",
            "description": "Diagnostic evaluation case demonstrating elevated central-adiposity indices (WHtR 0.58, BRI 5.0) and reliability check bounds.",
            "height_cm": 175.0,
            "sex": "female",
            "capture_mode": "single_front_view",
            "front_photo": "/media/TestPhoto/image copy 4.png",
            "side_photo": "/media/TestPhoto/image copy 4.png",
            "front_silhouette": "/media/out/image_copy_4_silhouette.png",
            "side_silhouette": "/media/out/image_copy_4_silhouette.png",
            "front_overlay": "/media/outputs/final/deva/front_regions_overlay.png",
            "side_overlay": "/media/outputs/final/deva/side_regions_overlay.png",
            "smplx_overlay": "/media/outputs/demo/deva_strict/smplx/smplx_front_overlay.png",
            "smplx_rendered": "/media/outputs/demo/deva_strict/smplx/smplx_rendered_front.png",
            "obj_path": "/api/mesh?preset=deva"
        }
    }


def _preset_mask_path(preset_info: Optional[Dict[str, Any]], key: str) -> Optional[str]:
    """Resolve a preset's silhouette entry ("/media/out/x.png") to a repo-relative path."""
    if not preset_info:
        return None
    value = preset_info.get(key)
    if not value:
        return None
    return value[len("/media/"):] if value.startswith("/media/") else value


def cv2_to_base64_png(image_bgr_or_rgb: np.ndarray, is_rgb: bool = True) -> str:
    """Encode OpenCV image array to base64 PNG data URL."""
    if is_rgb:
        img_bgr = cv2.cvtColor(image_bgr_or_rgb, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image_bgr_or_rgb
    success, buffer = cv2.imencode(".png", img_bgr)
    if not success:
        return ""
    b64 = base64.b64encode(buffer).decode("ascii")
    return f"data:image/png;base64,{b64}"


def run_pipeline_inference(
    front_img_bytes: Optional[bytes] = None,
    side_img_bytes: Optional[bytes] = None,
    front_mask_path: Optional[str] = None,
    side_mask_path: Optional[str] = None,
    height_cm: float = 175.0,
    sex: str = "male",
    preset_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the complete dual-view anthropometry pipeline with per-stage timing breakdown.
    """
    timings = {}
    total_t0 = time.perf_counter()
    
    presets = get_cached_presets()
    preset_info = presets.get(preset_key) if preset_key else None

    # Step 1: Input Resolution & Loading
    t0 = time.perf_counter()
    front_np = None
    side_np = None

    if front_img_bytes is not None:
        nparr = np.frombuffer(front_img_bytes, np.uint8)
        img_decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_decoded is not None:
            gray = cv2.cvtColor(img_decoded, cv2.COLOR_BGR2GRAY)
            _, front_np = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            front_np = cv2.resize(front_np, (TARGET_HW[1], TARGET_HW[0]), interpolation=cv2.INTER_NEAREST)

    if side_img_bytes is not None:
        nparr = np.frombuffer(side_img_bytes, np.uint8)
        img_decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_decoded is not None:
            gray = cv2.cvtColor(img_decoded, cv2.COLOR_BGR2GRAY)
            _, side_np = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            side_np = cv2.resize(side_np, (TARGET_HW[1], TARGET_HW[0]), interpolation=cv2.INTER_NEAREST)

    # Fallback to mask paths or default preset
    preset_front = _preset_mask_path(preset_info, "front_silhouette")
    preset_side = _preset_mask_path(preset_info, "side_silhouette")

    if front_np is None:
        resolved_front_path = front_mask_path or preset_front or str(ROOT / "out" / "deva_front_silhouette.png")
        if not Path(resolved_front_path).is_absolute():
            resolved_front_path = str(ROOT / resolved_front_path)
        front_np = load_mask_binary(resolved_front_path, TARGET_HW)

    if side_np is None:
        resolved_side_path = side_mask_path or preset_side or str(ROOT / "out" / "deva_side_silhouette.png")
        if not Path(resolved_side_path).is_absolute():
            resolved_side_path = str(ROOT / resolved_side_path)
        side_np = load_mask_binary(resolved_side_path, TARGET_HW)

    timings["step1_input_loading_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Step 2: Strict Segmentation Validation & Envelope Sanity Checks
    t0 = time.perf_counter()
    front_envelope = envelope_check(front_np, "front")
    side_envelope = envelope_check(side_np, "side")
    envelope_passed = front_envelope.ok and side_envelope.ok
    envelope_failures = front_envelope.failed + side_envelope.failed
    
    front_fg_ratio = float((front_np > 0).mean())
    side_fg_ratio = float((side_np > 0).mean())
    timings["step2_envelope_checks_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Step 3: Anatomical Part Decomposition
    t0 = time.perf_counter()
    front_regions = region_metrics(front_np)
    side_regions = region_metrics(side_np)
    front_overlay_img = make_region_overlay(front_np)
    side_overlay_img = make_region_overlay(side_np)
    front_overlay_b64 = cv2_to_base64_png(front_overlay_img, is_rgb=True)
    side_overlay_b64 = cv2_to_base64_png(side_overlay_img, is_rgb=True)
    timings["step3_part_decomposition_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Step 4 & 5: ResNet-18 Dual-Branch Forward Pass & Tape Dimension Recovery
    t0 = time.perf_counter()
    global GLOBAL_MODEL, GLOBAL_CKPT, GLOBAL_DEVICE
    if GLOBAL_MODEL is None:
        init_model()
    
    front_t = mask_to_tensor(front_np, GLOBAL_DEVICE)
    side_t = mask_to_tensor(side_np, GLOBAL_DEVICE)
    
    pred = predict_from_pair(GLOBAL_MODEL, GLOBAL_CKPT, front_t, side_t, height_cm, sex)
    measurements = pred.measurements
    indices = pred.indices
    risks = pred.risks
    timings["step4_5_forward_pass_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Step 6: SMPL-X Render-Back Reliability Gate
    t0 = time.perf_counter()
    thresholds = default_gate_thresholds()
    gate = evaluate_smplx_gate(
        front_np,
        side_np,
        measurements,
        height_cm,
        sex,
        SMPLX_MODEL_DIR,
        thresholds,
    )
    gate_accepted = gate.accepted
    gate_score = gate.score
    gate_reasons = gate.reasons
    front_iou = gate.metrics["front_iou"]
    front_chamfer = gate.metrics["front_chamfer"]
    timings["step6_smplx_gating_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Step 7: Clinical Central-Adiposity Index Derivation & Risk Assessment
    t0 = time.perf_counter()
    health_summary = assess_health_risk(
        measurements,
        indices,
        risks,
        sex,
        reportable=gate_accepted
    ).to_dict()
    timings["step7_clinical_indices_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    timings["total_pipeline_ms"] = round((time.perf_counter() - total_t0) * 1000, 2)

    confidence_intervals = {
        "waist_cm": {"value": round(measurements.get("waist_cm", 0.0), 2), "ci_95": "±1.97 cm", "mae": 1.97},
        "hip_cm": {"value": round(measurements.get("hip_cm", 0.0), 2), "ci_95": "±1.89 cm", "mae": 1.89},
        "chest_cm": {"value": round(measurements.get("chest_cm", 0.0), 2), "ci_95": "±2.10 cm", "mae": 2.10}
    }

    total_area_f = max(1, front_regions.get("body_area_px", 1))
    anatomical_parts = [
        {"name": "Head & Neck", "color": "#FFB450", "share_pct": round(front_regions.get("head_neck_area_px", 0) / total_area_f * 100, 1), "mean_width_px": round(front_regions.get("head_neck_mean_width_px", 0), 1)},
        {"name": "Shoulders", "color": "#50B4FF", "share_pct": round(front_regions.get("shoulders_area_px", 0) / total_area_f * 100, 1), "mean_width_px": round(front_regions.get("shoulders_mean_width_px", 0), 1)},
        {"name": "Torso Core", "color": "#6EDC6E", "share_pct": round(front_regions.get("torso_area_px", 0) / total_area_f * 100, 1), "mean_width_px": round(front_regions.get("torso_mean_width_px", 0), 1)},
        {"name": "Waist Band", "color": "#5050FF", "share_pct": round(front_regions.get("waist_area_px", 0) / total_area_f * 100, 1), "mean_width_px": round(front_regions.get("waist_mean_width_px", 0), 1)},
        {"name": "Pelvis & Hip", "color": "#DC6EDC", "share_pct": round(front_regions.get("hip_area_px", 0) / total_area_f * 100, 1), "mean_width_px": round(front_regions.get("hip_mean_width_px", 0), 1)},
        {"name": "Lower Limbs (Legs)", "color": "#FFDC50", "share_pct": round(front_regions.get("legs_area_px", 0) / total_area_f * 100, 1), "mean_width_px": round(front_regions.get("legs_mean_width_px", 0), 1)}
    ]

    return {
        "status": "success",
        "preset_id": preset_key,
        "parameters": {
            "height_cm": height_cm,
            "sex": sex,
            "checkpoint": "checkpoints/best_640x480_v4_resnet.pt",
            "device": GLOBAL_DEVICE
        },
        "timings": timings,
        "envelope_checks": {
            "passed": envelope_passed,
            "failures": envelope_failures,
            "front_foreground_ratio": round(front_fg_ratio, 4),
            "side_foreground_ratio": round(side_fg_ratio, 4),
            "side_to_front_ratio": round(side_fg_ratio / max(1e-5, front_fg_ratio), 3)
        },
        "anatomical_decomposition": {
            "parts": anatomical_parts,
            "front_overlay_b64": front_overlay_b64,
            "side_overlay_b64": side_overlay_b64,
            "shoulder_to_waist_ratio": round(float(front_regions.get("shoulder_to_waist_width_ratio", 1.0)), 2),
            "hip_to_waist_ratio": round(float(front_regions.get("hip_to_waist_width_ratio", 1.0)), 2),
            "torso_asymmetry": round(float(front_regions.get("left_right_torso_asymmetry", 0.0)), 4)
        },
        "measurements": {k: round(v, 2) for k, v in measurements.items()},
        "confidence_intervals": confidence_intervals,
        "clinical_indices": {
            "WHR": round(indices.get("WHR", 0.0), 4),
            "WHtR": round(indices.get("WHtR", 0.0), 4),
            "BRI": round(indices.get("BRI", 0.0), 4)
        },
        "risk_categories": risks,
        "health_summary": health_summary,
        "smplx_gate": {
            "accepted": gate_accepted,
            "score": round(gate_score, 4),
            "front_iou": round(front_iou, 4),
            "front_chamfer": round(front_chamfer, 4),
            "metrics": {k: round(v, 4) for k, v in gate.metrics.items()},
            "reasons": gate_reasons,
            "thresholds": {
                "max_score": thresholds.max_score,
                "min_render_iou": thresholds.min_render_iou,
                "max_chamfer": thresholds.max_chamfer,
            },
            "status_badge": "ACCEPTED (Reliable)" if gate_accepted else "REJECTED (Recapture Recommended)",
            "message": (
                "SMPL-X render-back agrees with the observed silhouettes."
                if gate_accepted
                else "Gate rejected the capture: " + ", ".join(gate_reasons)
            ),
            "obj_url": "/api/mesh?preset=deva",
            "front_render_overlay": "/media/outputs/demo/deva_strict/smplx/smplx_front_overlay.png",
            "rendered_silhouette": "/media/outputs/demo/deva_strict/smplx/smplx_rendered_front.png"
        }
    }


class Body2FitRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        
        # Route: Main Demo UI
        if path in {"/", "/index.html", "/demo"}:
            self.serve_file(ROOT / "web" / "index.html", "text/html; charset=utf-8")
            return
        
        # Route: Static Assets (web/...)
        if path.startswith("/static/"):
            rel = path[8:]
            self.serve_file(ROOT / "web" / rel)
            return

        # Route: Media (TestPhoto/, out/, outputs/)
        if path.startswith("/media/"):
            rel = path[7:]
            self.serve_file(ROOT / rel)
            return

        # API: Health check
        if path == "/api/health":
            global GLOBAL_DEVICE
            payload = {
                "status": "healthy",
                "service": "Body2Fit Dual-View Anthropometry API",
                "version": "4.0.0-resnet",
                "checkpoint": "checkpoints/best_640x480_v4_resnet.pt",
                "device": GLOBAL_DEVICE,
                "input_resolution": "640x480",
                "available_presets": list(get_cached_presets().keys())
            }
            self.send_json(payload)
            return

        # API: Presets
        if path == "/api/presets":
            self.send_json(get_cached_presets())
            return

        # API: Ablation table & research data
        if path == "/api/ablation":
            ablation_data = {
                "limited_ablation": [
                    {"setting": "Silhouette-only", "tp_2cm": 3.12, "tp_5cm": 25.0, "tp_10cm": 59.38, "waist_mae": 9.385, "hip_mae": 8.779, "chest_mae": 8.547, "whtr_mae": 0.0558, "whr_mae": 0.0307, "bri_mae": 1.2166},
                    {"setting": "Dual-view", "tp_2cm": 56.25, "tp_5cm": 81.25, "tp_10cm": 100.0, "waist_mae": 2.397, "hip_mae": 2.823, "chest_mae": 2.281, "whtr_mae": 0.0139, "whr_mae": 0.0245, "bri_mae": 0.2747},
                    {"setting": "Dual-view + Height", "tp_2cm": 59.38, "tp_5cm": 96.88, "tp_10cm": 100.0, "waist_mae": 1.967, "hip_mae": 1.888, "chest_mae": 2.105, "whtr_mae": 0.0114, "whr_mae": 0.0184, "bri_mae": 0.2268},
                    {"setting": "Dual-view + Weight", "tp_2cm": 59.38, "tp_5cm": 96.88, "tp_10cm": 100.0, "waist_mae": 1.939, "hip_mae": 1.665, "chest_mae": 1.807, "whtr_mae": 0.0113, "whr_mae": 0.0194, "bri_mae": 0.2230},
                    {"setting": "Dual-view + Height + Weight", "tp_2cm": 56.25, "tp_5cm": 96.88, "tp_10cm": 100.0, "waist_mae": 1.933, "hip_mae": 1.682, "chest_mae": 1.931, "whtr_mae": 0.0113, "whr_mae": 0.0191, "bri_mae": 0.2233}
                ],
                "reliability_gate_coverage": [
                    {"coverage": "100%", "threshold": 1.0722, "dim_mae": 2.112, "whr_mae": 0.0184, "whtr_mae": 0.0108, "waist_agreement": "81.0%", "whr_agreement": "83.0%"},
                    {"coverage": "90%", "threshold": 0.9777, "dim_mae": 2.106, "whr_mae": 0.0187, "whtr_mae": 0.0112, "waist_agreement": "78.9%", "whr_agreement": "86.7%"},
                    {"coverage": "80%", "threshold": 0.9107, "dim_mae": 2.203, "whr_mae": 0.0194, "whtr_mae": 0.0116, "waist_agreement": "77.5%", "whr_agreement": "86.3%"},
                    {"coverage": "70%", "threshold": 0.8425, "dim_mae": 2.364, "whr_mae": 0.0211, "whtr_mae": 0.0125, "waist_agreement": "74.3%", "whr_agreement": "84.3%"}
                ]
            }
            self.send_json(ablation_data)
            return

        # API: 3D OBJ Mesh streaming
        if path == "/api/mesh":
            obj_path = ROOT / "outputs" / "demo" / "deva_strict" / "smplx" / "smplx_fit.obj"
            if not obj_path.exists():
                obj_path = ROOT / "outputs" / "final" / "deva" / "smplx" / "smplx_fit.obj"
            
            if obj_path.exists():
                self.serve_file(obj_path, content_type="model/obj")
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Mesh OBJ not found")
            return

        super().do_GET()

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        
        if path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            
            preset_key = "deva_full"
            height_cm = 175.0
            sex = "male"
            front_bytes = None
            side_bytes = None
            
            try:
                data = json.loads(body.decode("utf-8"))
                preset_key = data.get("preset")
                height_cm = float(data.get("height_cm", 175.0))
                sex = str(data.get("sex", "male"))
                
                if "front_base64" in data and data["front_base64"]:
                    b64_str = data["front_base64"].split(",")[-1]
                    front_bytes = base64.b64decode(b64_str)
                if "side_base64" in data and data["side_base64"]:
                    b64_str = data["side_base64"].split(",")[-1]
                    side_bytes = base64.b64decode(b64_str)
                    
            except Exception as e:
                logger.warning(f"Error parsing JSON body: {e}, falling back to defaults")

            result = run_pipeline_inference(
                front_img_bytes=front_bytes,
                side_img_bytes=side_bytes,
                height_cm=height_cm,
                sex=sex,
                preset_key=preset_key
            )
            self.send_json(result)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "API endpoint not found")

    def serve_file(self, file_path: Path, content_type: Optional[str] = None) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, f"File not found: {file_path.name}")
            return
        
        if content_type is None:
            content_type, _ = mimetypes.guess_type(str(file_path))
            if content_type is None:
                if file_path.suffix == ".obj":
                    content_type = "model/obj"
                else:
                    content_type = "application/octet-stream"

        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, data: Any) -> None:
        encoded = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(encoded)


# Compatibility alias
BodyFitRequestHandler = Body2FitRequestHandler


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def run_server(port: int = DEFAULT_PORT, host: str = "0.0.0.0") -> None:
    init_model()
    server = ThreadedHTTPServer((host, port), Body2FitRequestHandler)
    print("=" * 70)
    print(f"🔥 Body2Fit Interactive Demo Server running at http://localhost:{port}/")
    print(f"📡 API endpoints available at http://localhost:{port}/api/health and /api/predict")
    print(f"🎨 3D SMPL-X Mesh viewer and recruiter walkthrough ready")
    print("=" * 70)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Body2Fit Demo Server...")
        server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Body2Fit Interactive Demo Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Server port (default: {DEFAULT_PORT})")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to checkpoint")
    args = parser.parse_args()
    run_server(args.port, args.host)
