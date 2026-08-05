from __future__ import annotations

import math
from typing import Dict


def _sex_key(sex: str | None) -> str:
    value = (sex or "").strip().lower()
    if value in {"male", "m", "man"}:
        return "male"
    if value in {"female", "f", "woman"}:
        return "female"
    return "unknown"


def whtr(waist_cm: float, height_cm: float) -> float:
    if height_cm <= 0.0:
        raise ValueError(f"height_cm must be positive, got {height_cm}")
    return waist_cm / height_cm


def whr(waist_cm: float, hip_cm: float) -> float:
    if hip_cm <= 0.0:
        raise ValueError(f"hip_cm must be positive, got {hip_cm}")
    return waist_cm / hip_cm


def bri(waist_cm: float, height_cm: float) -> float:
    if height_cm <= 0.0:
        raise ValueError(f"height_cm must be positive, got {height_cm}")
    waist_m = waist_cm / 100.0
    height_m = height_cm / 100.0
    denominator = (0.5 * height_m) ** 2
    if denominator <= 0.0:
        raise ValueError(f"height_cm produced a non-positive denominator, got {height_cm}")
    ratio = ((waist_m / (2.0 * math.pi)) ** 2) / denominator
    return 364.2 - (365.5 * math.sqrt(max(0.0, 1.0 - ratio)))


def derive_indices(measurements: Dict[str, float], height_cm: float | None) -> Dict[str, float]:
    indices: Dict[str, float] = {}
    waist = measurements.get("waist_cm")
    hip = measurements.get("hip_cm")

    if waist is not None and hip is not None:
        indices["WHR"] = whr(waist, hip)
    if waist is not None and height_cm is not None:
        indices["WHtR"] = whtr(waist, height_cm)
        indices["BRI"] = bri(waist, height_cm)

    return indices


def waist_risk_category(waist_cm: float, sex: str | None) -> str:
    if waist_cm <= 0.0:
        raise ValueError(f"waist_cm must be positive, got {waist_cm}")
    sex_norm = _sex_key(sex)
    if sex_norm == "male":
        if waist_cm >= 102.0:
            return "substantially_increased"
        if waist_cm >= 94.0:
            return "increased"
        return "not_increased"
    if sex_norm == "female":
        if waist_cm >= 88.0:
            return "substantially_increased"
        if waist_cm >= 80.0:
            return "increased"
        return "not_increased"
    return "sex_required"


def whr_risk_category(whr_value: float, sex: str | None) -> str:
    if whr_value <= 0.0:
        raise ValueError(f"WHR must be positive, got {whr_value}")
    sex_norm = _sex_key(sex)
    if sex_norm == "male":
        return "substantially_increased" if whr_value > 0.90 else "not_increased"
    if sex_norm == "female":
        return "substantially_increased" if whr_value > 0.85 else "not_increased"
    return "sex_required"


def whtr_risk_category(whtr_value: float) -> str:
    if whtr_value <= 0.0:
        raise ValueError(f"WHtR must be positive, got {whtr_value}")
    if whtr_value >= 0.60:
        return "high"
    if whtr_value >= 0.50:
        return "increased"
    return "not_increased"


def derive_risk_categories(
    measurements: Dict[str, float],
    indices: Dict[str, float],
    sex: str | None,
) -> Dict[str, str]:
    risks: Dict[str, str] = {}
    waist = measurements.get("waist_cm")
    if waist is not None:
        risks["waist_circumference"] = waist_risk_category(waist, sex)
    whr_value = indices.get("WHR")
    if whr_value is not None:
        risks["WHR"] = whr_risk_category(whr_value, sex)
    whtr_value = indices.get("WHtR")
    if whtr_value is not None:
        risks["WHtR_secondary"] = whtr_risk_category(whtr_value)
    if "BRI" in indices:
        risks["BRI"] = "exploratory_no_who_category"
    return risks
