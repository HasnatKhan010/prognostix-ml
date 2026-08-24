"""Translating a predicted RUL into a maintenance decision.

A number of cycles is not actionable on its own. A planner needs to know whether
this machine is fine, worth watching, or about to take the line down. This module
maps RUL onto a 0-100 health score and a risk band with an explicit recommended
action, using the thresholds under ``inference.health`` in the config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.config import Config, get_config

__all__ = [
    "HealthAssessment",
    "RiskLevel",
    "assess_health",
    "health_score",
    "risk_level",
]


class RiskLevel(str, Enum):
    """Operational risk bands, ordered from least to most urgent."""

    HEALTHY = "healthy"
    WATCH = "watch"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def severity(self) -> int:
        """0 (healthy) to 3 (critical), for sorting and alert routing."""
        return _SEVERITY[self]


_SEVERITY = {
    RiskLevel.HEALTHY: 0,
    RiskLevel.WATCH: 1,
    RiskLevel.WARNING: 2,
    RiskLevel.CRITICAL: 3,
}

RECOMMENDATIONS = {
    RiskLevel.HEALTHY: "No action required. Continue scheduled monitoring.",
    RiskLevel.WATCH: "Increase monitoring frequency and review sensor trends.",
    RiskLevel.WARNING: "Schedule maintenance within the next planned window.",
    RiskLevel.CRITICAL: "Take out of service and inspect immediately.",
}

DEFAULT_THRESHOLDS = {
    "max_rul": 125.0,
    "critical_rul": 20.0,
    "warning_rul": 50.0,
    "watch_rul": 80.0,
}


@dataclass
class HealthAssessment:
    """A prediction plus the decision that follows from it."""

    rul: float
    health_score: float
    risk_level: RiskLevel
    recommended_action: str
    thresholds: dict[str, float] = field(default_factory=dict)
    engine_id: int | str | None = None
    model: str | None = None

    @property
    def severity(self) -> int:
        return self.risk_level.severity

    @property
    def requires_action(self) -> bool:
        """True from the warning band upwards."""
        return self.risk_level.severity >= RiskLevel.WARNING.severity

    def to_dict(self) -> dict[str, object]:
        """Flat, JSON-serialisable representation."""
        payload: dict[str, object] = {
            "rul": round(float(self.rul), 2),
            "health_score": round(float(self.health_score), 2),
            "risk_level": self.risk_level.value,
            "recommended_action": self.recommended_action,
            "requires_action": self.requires_action,
        }
        if self.engine_id is not None:
            payload["engine_id"] = self.engine_id
        if self.model is not None:
            payload["model"] = self.model
        return payload


def _thresholds(config: Config | None) -> dict[str, float]:
    """Resolve health thresholds from config, falling back to defaults."""
    try:
        configured = (config or get_config()).inference.health
        values = {key: float(configured.get(key, default)) for key, default in DEFAULT_THRESHOLDS.items()}
    except Exception:
        values = dict(DEFAULT_THRESHOLDS)

    if not (values["critical_rul"] <= values["warning_rul"] <= values["watch_rul"]):
        raise ValueError(
            "Health thresholds must satisfy critical_rul <= warning_rul <= watch_rul, "
            f"got {values}"
        )
    if values["max_rul"] <= 0:
        raise ValueError(f"max_rul must be positive, got {values['max_rul']}")
    return values


def health_score(
    rul: float, max_rul: float | None = None, config: Config | None = None
) -> float:
    """Map RUL to a 0-100 health score.

    The scale is linear in remaining cycles and saturates at ``max_rul``: an
    engine with 300 cycles left is not "more healthy" than one with 130, because
    degradation is not yet observable in either.
    """
    thresholds = _thresholds(config)
    ceiling = float(max_rul if max_rul is not None else thresholds["max_rul"])
    if ceiling <= 0:
        raise ValueError(f"max_rul must be positive, got {ceiling}")
    return float(min(max(float(rul), 0.0) / ceiling, 1.0) * 100.0)


def risk_level(rul: float, config: Config | None = None) -> RiskLevel:
    """Bucket a RUL prediction into a :class:`RiskLevel`."""
    thresholds = _thresholds(config)
    value = float(rul)
    if value <= thresholds["critical_rul"]:
        return RiskLevel.CRITICAL
    if value <= thresholds["warning_rul"]:
        return RiskLevel.WARNING
    if value <= thresholds["watch_rul"]:
        return RiskLevel.WATCH
    return RiskLevel.HEALTHY


def assess_health(
    rul: float,
    config: Config | None = None,
    engine_id: int | str | None = None,
    model: str | None = None,
) -> HealthAssessment:
    """Build a full assessment - score, band and recommended action - from a RUL."""
    thresholds = _thresholds(config)
    level = risk_level(rul, config)
    return HealthAssessment(
        rul=max(float(rul), 0.0),
        health_score=health_score(rul, thresholds["max_rul"], config),
        risk_level=level,
        recommended_action=RECOMMENDATIONS[level],
        thresholds=thresholds,
        engine_id=engine_id,
        model=model,
    )
