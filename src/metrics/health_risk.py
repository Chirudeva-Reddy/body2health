from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


HEALTH_RISK_DISCLAIMER = (
    "Screening estimate of central-adiposity and cardiometabolic risk from image-based "
    "anthropometry; not a diagnosis or substitute for clinical assessment."
)


@dataclass(frozen=True)
class RiskComponent:
    name: str
    value: float | None
    category: str
    label: str
    borderline: bool
    message: str


@dataclass(frozen=True)
class HealthSummary:
    overall_risk: str
    primary_driver: str
    reportable: bool
    message: str
    disclaimer: str
    components: List[RiskComponent]

    def to_dict(self) -> Dict[str, object]:
        return {
            "overall_risk": self.overall_risk,
            "primary_driver": self.primary_driver,
            "reportable": self.reportable,
            "message": self.message,
            "disclaimer": self.disclaimer,
            "components": [asdict(component) for component in self.components],
        }


def assess_health_risk(
    measurements: Dict[str, float],
    indices: Dict[str, float],
    risks: Dict[str, str],
    sex: str | None,
    reportable: bool,
) -> HealthSummary:
    components = [
        _whtr_component(indices),
        _whr_component(indices, risks, sex),
        _waist_component(measurements, risks, sex),
        _bri_component(indices),
    ]
    if not reportable:
        return HealthSummary(
            overall_risk="not_reported",
            primary_driver="geometry_gate",
            reportable=False,
            message=(
                "Central-adiposity risk labels were not reported because the SMPL-X "
                "reliability gate rejected the capture."
            ),
            disclaimer=HEALTH_RISK_DISCLAIMER,
            components=components,
        )

    ranked_components = [
        component
        for component in components
        if component.category not in {"exploratory", "missing", "sex_required"}
    ]
    if len(ranked_components) == 0:
        return HealthSummary(
            overall_risk="not_reported",
            primary_driver="missing_inputs",
            reportable=False,
            message="Central-adiposity risk labels need waist, hip, height, and sex inputs.",
            disclaimer=HEALTH_RISK_DISCLAIMER,
            components=components,
        )

    primary = max(ranked_components, key=lambda component: _severity(component.category))
    overall = _overall_category(primary.category)
    return HealthSummary(
        overall_risk=overall,
        primary_driver=primary.name,
        reportable=True,
        message=_summary_message(primary, overall),
        disclaimer=HEALTH_RISK_DISCLAIMER,
        components=components,
    )


def _whtr_component(indices: Dict[str, float]) -> RiskComponent:
    value = indices.get("WHtR")
    if value is None:
        return _missing_component("WHtR", "WHtR requires waist and height.")
    if value >= 0.60:
        category = "high"
        label = "High central-adiposity screening risk"
    elif value >= 0.50:
        category = "increased"
        label = "Increased central-adiposity screening risk"
    else:
        category = "not_increased"
        label = "No increased WHtR central-adiposity risk"
    borderline = _near_any(value, (0.50, 0.60), 0.02)
    return RiskComponent(
        name="WHtR",
        value=value,
        category=category,
        label=label,
        borderline=borderline,
        message=_threshold_message("WHtR", value, "0.50 and 0.60", borderline),
    )


def _whr_component(
    indices: Dict[str, float],
    risks: Dict[str, str],
    sex: str | None,
) -> RiskComponent:
    value = indices.get("WHR")
    if value is None:
        return _missing_component("WHR", "WHR requires waist and hip.")
    threshold = _whr_threshold(sex)
    category = risks.get("WHR", "sex_required")
    if threshold is None:
        return RiskComponent(
            name="WHR",
            value=value,
            category="sex_required",
            label="Sex required for WHR risk",
            borderline=False,
            message="WHR risk thresholds require sex-specific interpretation.",
        )
    normalized = _normalize_category(category)
    borderline = abs(value - threshold) <= 0.02
    return RiskComponent(
        name="WHR",
        value=value,
        category=normalized,
        label=_label_for_category(normalized, "WHR"),
        borderline=borderline,
        message=_threshold_message("WHR", value, f"{threshold:.2f}", borderline),
    )


def _waist_component(
    measurements: Dict[str, float],
    risks: Dict[str, str],
    sex: str | None,
) -> RiskComponent:
    value = measurements.get("waist_cm")
    if value is None:
        return _missing_component("waist_circumference", "Waist risk requires waist_cm.")
    thresholds = _waist_thresholds(sex)
    category = risks.get("waist_circumference", "sex_required")
    if thresholds is None:
        return RiskComponent(
            name="waist_circumference",
            value=value,
            category="sex_required",
            label="Sex required for waist risk",
            borderline=False,
            message="Waist circumference risk thresholds require sex-specific interpretation.",
        )
    normalized = _normalize_category(category)
    borderline = _near_any(value, thresholds, 2.0)
    threshold_text = " and ".join(f"{threshold:.0f} cm" for threshold in thresholds)
    return RiskComponent(
        name="waist_circumference",
        value=value,
        category=normalized,
        label=_label_for_category(normalized, "waist circumference"),
        borderline=borderline,
        message=_threshold_message("waist circumference", value, threshold_text, borderline),
    )


def _bri_component(indices: Dict[str, float]) -> RiskComponent:
    value = indices.get("BRI")
    if value is None:
        return _missing_component("BRI", "BRI requires waist and height.")
    return RiskComponent(
        name="BRI",
        value=value,
        category="exploratory",
        label="Exploratory index",
        borderline=False,
        message="BRI is reported for context only; no WHO-style risk category is assigned.",
    )


def _summary_message(primary: RiskComponent, overall: str) -> str:
    if primary.borderline:
        return (
            f"Estimated central-adiposity/cardiometabolic screening risk is {overall}, "
            f"driven by {primary.name}. The value is close to a screening cutoff, so "
            "recapture or tape measurement confirmation is recommended."
        )
    return (
        f"Estimated central-adiposity/cardiometabolic screening risk is {overall}, "
        f"driven by {primary.name}."
    )


def _threshold_message(name: str, value: float, threshold_text: str, borderline: bool) -> str:
    if borderline:
        return f"{name} is {value:.4f}, close to cutoff {threshold_text}."
    return f"{name} is {value:.4f}; cutoff reference: {threshold_text}."


def _missing_component(name: str, message: str) -> RiskComponent:
    return RiskComponent(
        name=name,
        value=None,
        category="missing",
        label="Missing input",
        borderline=False,
        message=message,
    )


def _whr_threshold(sex: str | None) -> float | None:
    value = (sex or "").strip().lower()
    if value in {"male", "m", "man"}:
        return 0.90
    if value in {"female", "f", "woman"}:
        return 0.85
    return None


def _waist_thresholds(sex: str | None) -> tuple[float, float] | None:
    value = (sex or "").strip().lower()
    if value in {"male", "m", "man"}:
        return 94.0, 102.0
    if value in {"female", "f", "woman"}:
        return 80.0, 88.0
    return None


def _near_any(value: float, thresholds: tuple[float, ...], tolerance: float) -> bool:
    return any(abs(value - threshold) <= tolerance for threshold in thresholds)


def _normalize_category(category: str) -> str:
    if category == "substantially_increased":
        return "high"
    return category


def _overall_category(category: str) -> str:
    if category == "substantially_increased":
        return "high"
    if category in {"not_increased", "increased", "high"}:
        return category
    return "not_reported"


def _label_for_category(category: str, metric_name: str) -> str:
    if category == "high":
        return f"High {metric_name} risk"
    if category == "increased":
        return f"Increased {metric_name} risk"
    if category == "not_increased":
        return f"No increased {metric_name} risk"
    if category == "sex_required":
        return f"Sex required for {metric_name} risk"
    return f"{metric_name} risk not reported"


def _severity(category: str) -> int:
    normalized = _normalize_category(category)
    if normalized == "high":
        return 3
    if normalized == "increased":
        return 2
    if normalized == "not_increased":
        return 1
    return 0
