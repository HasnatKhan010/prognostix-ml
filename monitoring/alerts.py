"""Alert creation, routing and persistence.

Detectors in this package do not decide what happens next - they hand an
:class:`Alert` to an :class:`AlertManager`, which appends it to a JSONL log,
optionally prints it and optionally POSTs it to a webhook. Keeping the sinks
here means adding Slack or PagerDuty later touches one file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from src.config import Config, get_config

logger = logging.getLogger(__name__)

__all__ = ["Alert", "AlertManager", "Severity", "alert_from_assessment"]


class Severity(str, Enum):
    """How urgently a human needs to look."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "critical": 2}[self.value]


@dataclass
class Alert:
    """A single actionable event."""

    title: str
    message: str
    severity: Severity = Severity.INFO
    source: str = "prognostix"
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None
    entity: str | int | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form (the enum becomes its string value)."""
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return payload

    def format(self) -> str:
        """One-line human-readable rendering."""
        parts = [f"[{self.severity.value.upper()}] {self.title}"]
        if self.entity is not None:
            parts.append(f"entity={self.entity}")
        if self.metric and self.value is not None:
            measured = f"{self.metric}={self.value:.4g}"
            if self.threshold is not None:
                measured += f" (threshold {self.threshold:.4g})"
            parts.append(measured)
        parts.append(self.message)
        return " | ".join(parts)


class AlertManager:
    """Fan an alert out to the configured sinks.

    Parameters
    ----------
    log_path:
        JSONL file alerts are appended to. Defaults to
        ``artifacts/reports/alerts.jsonl``.
    console:
        Also emit through the logger at a level matching the severity.
    webhook_url:
        POST target. Failures are logged, never raised - a broken webhook must
        not take down the job that raised the alert.
    min_severity:
        Alerts below this rank are dropped.
    """

    def __init__(
        self,
        config: Config | None = None,
        log_path: str | Path | None = None,
        console: bool | None = None,
        webhook_url: str | None = None,
        min_severity: Severity = Severity.INFO,
    ):
        self.config = config or get_config()
        settings = self.config.monitoring.get("alerts", {}) or {}

        if log_path is None:
            filename = settings.get("log_filename", "alerts.jsonl")
            log_path = self.config.path("reports") / filename
        self.log_path = Path(log_path)
        self.console = settings.get("console", True) if console is None else console
        self.webhook_url = webhook_url or settings.get("webhook_url")
        self.min_severity = min_severity
        self.sent: list[Alert] = []

    def emit(self, alert: Alert) -> bool:
        """Record and route one alert. Returns False when filtered out."""
        if alert.severity.rank < self.min_severity.rank:
            return False

        self.sent.append(alert)
        self._write(alert)
        if self.console:
            level = {
                Severity.INFO: logging.INFO,
                Severity.WARNING: logging.WARNING,
                Severity.CRITICAL: logging.ERROR,
            }[alert.severity]
            logger.log(level, alert.format())
        if self.webhook_url:
            self._post(alert)
        return True

    def emit_many(self, alerts: Iterable[Alert]) -> int:
        """Route several alerts; returns how many were sent."""
        return sum(1 for alert in alerts if self.emit(alert))

    def _write(self, alert: Alert) -> None:
        """Append the alert to the JSONL log."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(alert.to_dict()) + "\n")
        except OSError as exc:  # disk problems must not mask the alert itself
            logger.error("Could not write alert to %s: %s", self.log_path, exc)

    def _post(self, alert: Alert) -> None:
        """Best-effort webhook delivery."""
        try:
            import requests

            response = requests.post(
                self.webhook_url,
                json=alert.to_dict(),
                timeout=5,
            )
            if response.status_code >= 400:
                logger.warning(
                    "Webhook returned %s: %s", response.status_code, response.text[:200]
                )
        except Exception as exc:  # noqa: BLE001 - never propagate sink failures
            logger.warning("Webhook delivery failed: %s", exc)

    def history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Read alerts back from the JSONL log, newest last."""
        if not self.log_path.exists():
            return []
        with self.log_path.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        return records[-limit:] if limit else records

    def summary(self) -> dict[str, int]:
        """Count the alerts emitted by this manager, by severity."""
        counts = {severity.value: 0 for severity in Severity}
        for alert in self.sent:
            counts[alert.severity.value] += 1
        return counts


def alert_from_assessment(assessment, threshold_key: str = "warning_rul") -> Alert | None:
    """Build an alert from a :class:`~src.inference.health_score.HealthAssessment`.

    Returns ``None`` for machines that need no action, so callers can map over a
    whole fleet and filter.
    """
    from src.inference.health_score import RiskLevel

    if not assessment.requires_action:
        return None

    severity = (
        Severity.CRITICAL
        if assessment.risk_level is RiskLevel.CRITICAL
        else Severity.WARNING
    )
    return Alert(
        title=f"{assessment.risk_level.value.title()} RUL for engine {assessment.engine_id}",
        message=assessment.recommended_action,
        severity=severity,
        source="rul_monitor",
        metric="rul_cycles",
        value=float(assessment.rul),
        threshold=float(assessment.thresholds.get(threshold_key, 0.0)),
        entity=assessment.engine_id,
        metadata={
            "health_score": round(float(assessment.health_score), 2),
            "model": assessment.model,
        },
    )
