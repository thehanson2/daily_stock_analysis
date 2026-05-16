# -*- coding: utf-8 -*-
"""Service layer for Alert API MVP."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, Optional

from src.agent.events import (
    EventMonitor,
    PriceAlert,
    PriceChangeAlert,
    VolumeAlert,
    _read_quote_float,
    validate_event_alert_rule,
)
from src.repositories.alert_repo import AlertRepository
from src.storage import AlertNotificationRecord, AlertRuleRecord, AlertTriggerRecord, DatabaseManager


SUPPORTED_ALERT_TYPES = frozenset({"price_cross", "price_change_percent", "volume_spike"})
SUPPORTED_TARGET_SCOPES = frozenset({"single_symbol"})
SUPPORTED_SEVERITIES = frozenset({"info", "warning", "critical"})
NULLABLE_RULE_UPDATE_FIELDS = frozenset({"cooldown_policy", "notification_policy"})


class AlertServiceError(ValueError):
    """Raised when alert service input is invalid."""

    error_code = "validation_error"


class AlertNotFoundError(AlertServiceError):
    """Raised when an alert resource does not exist."""

    error_code = "not_found"


class UnsupportedAlertTypeError(AlertServiceError):
    """Raised when the API receives a future/non-runtime alert type."""

    error_code = "unsupported_alert_type"


class AlertService:
    """Business logic for alert rule CRUD and dry-run evaluation."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager.get_instance()
        self.repo = AlertRepository(self.db)

    def create_rule(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        fields = self._normalize_rule_payload(payload)
        return self._serialize_rule(self.repo.create_rule(fields))

    def get_rule(self, rule_id: int) -> Dict[str, Any]:
        row = self.repo.get_rule(rule_id)
        if row is None:
            raise AlertNotFoundError(f"Alert rule not found: {rule_id}")
        return self._serialize_rule(row)

    def update_rule(self, rule_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        row = self.repo.get_rule(rule_id)
        if row is None:
            raise AlertNotFoundError(f"Alert rule not found: {rule_id}")
        if not payload:
            raise AlertServiceError("No fields provided for update")
        self._validate_rule_update_payload(payload)

        merged = self._serialize_rule(row)
        merged.update(payload)
        fields = self._normalize_rule_payload(merged, source=merged.get("source") or "api")
        updated = self.repo.update_rule(rule_id, fields)
        if updated is None:
            raise AlertNotFoundError(f"Alert rule not found: {rule_id}")
        return self._serialize_rule(updated)

    def delete_rule(self, rule_id: int) -> bool:
        return self.repo.delete_rule(rule_id)

    def enable_rule(self, rule_id: int, enabled: bool) -> Dict[str, Any]:
        updated = self.repo.update_rule(rule_id, {"enabled": enabled})
        if updated is None:
            raise AlertNotFoundError(f"Alert rule not found: {rule_id}")
        return self._serialize_rule(updated)

    def list_rules(
        self,
        *,
        enabled: Optional[bool] = None,
        alert_type: Optional[str] = None,
        target_scope: Optional[str] = None,
        target: Optional[str] = None,
        source: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        rows, total = self.repo.list_rules(
            enabled=enabled,
            alert_type=alert_type,
            target_scope=target_scope,
            target=target,
            source=source,
            page=page,
            page_size=page_size,
        )
        return {
            "items": [self._serialize_rule(row) for row in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def test_rule(self, rule_id: int) -> Dict[str, Any]:
        row = self.repo.get_rule(rule_id)
        if row is None:
            raise AlertNotFoundError(f"Alert rule not found: {rule_id}")

        rule = self._to_runtime_rule(row)
        monitor = EventMonitor()
        try:
            return asyncio.run(self._evaluate_rule(rule, monitor))
        except Exception as exc:
            return {
                "rule_id": rule_id,
                "status": "evaluation_error",
                "triggered": False,
                "observed_value": None,
                "message": self._sanitize_text(str(exc) or "Alert evaluation failed"),
            }

    async def _evaluate_rule(self, rule, monitor: EventMonitor) -> Dict[str, Any]:
        if isinstance(rule, PriceAlert):
            return await self._evaluate_price(rule, monitor)
        if isinstance(rule, PriceChangeAlert):
            return await self._evaluate_price_change(rule, monitor)
        if isinstance(rule, VolumeAlert):
            return await self._evaluate_volume(rule)
        return self._evaluation_error(rule, f"unsupported runtime alert type: {rule.alert_type}")

    async def _evaluate_price(self, rule: PriceAlert, monitor: EventMonitor) -> Dict[str, Any]:
        try:
            quote = await monitor._get_realtime_quote(rule.stock_code)
        except Exception as exc:
            return self._evaluation_error(rule, exc)
        if quote is None:
            return self._not_triggered(rule, None, "No realtime quote available")

        try:
            current_price = float(getattr(quote, "price", 0) or 0)
        except (TypeError, ValueError) as exc:
            return self._evaluation_error(rule, exc)
        if current_price <= 0:
            return self._not_triggered(rule, None, "No valid realtime price available")

        triggered = (
            (rule.direction == "above" and current_price >= rule.price)
            or (rule.direction == "below" and current_price <= rule.price)
        )
        if triggered:
            return self._triggered(
                rule,
                current_price,
                f"{rule.stock_code} price {rule.direction} {rule.price}: current = {current_price}",
            )
        return self._not_triggered(
            rule,
            current_price,
            f"{rule.stock_code} price {current_price} did not cross {rule.direction} {rule.price}",
        )

    async def _evaluate_price_change(self, rule: PriceChangeAlert, monitor: EventMonitor) -> Dict[str, Any]:
        try:
            quote = await monitor._get_realtime_quote(rule.stock_code)
        except Exception as exc:
            return self._evaluation_error(rule, exc)
        if quote is None:
            return self._not_triggered(rule, None, "No realtime quote available")

        current_change_pct = _read_quote_float(
            quote,
            "change_pct",
            "change_percent",
            "pct_chg",
            "change_rate",
        )
        if current_change_pct is None:
            return self._not_triggered(rule, None, "No valid realtime change percent available")

        threshold = abs(float(rule.change_pct))
        direction = rule.direction.lower()
        triggered = (
            (direction == "up" and current_change_pct >= threshold)
            or (direction == "down" and current_change_pct <= -threshold)
        )
        if triggered:
            return self._triggered(
                rule,
                current_change_pct,
                f"{rule.stock_code} change {direction} {threshold:.2f}%: current = {current_change_pct:+.2f}%",
            )
        return self._not_triggered(
            rule,
            current_change_pct,
            f"{rule.stock_code} change {current_change_pct:+.2f}% did not cross {direction} {threshold:.2f}%",
        )

    async def _evaluate_volume(self, rule: VolumeAlert) -> Dict[str, Any]:
        def _fetch_daily_data():
            from data_provider import DataFetcherManager

            return DataFetcherManager().get_daily_data(rule.stock_code, days=20)

        try:
            result = await asyncio.to_thread(_fetch_daily_data)
        except Exception as exc:
            return self._evaluation_error(rule, exc)
        if result is None:
            return self._not_triggered(rule, None, "No daily volume data available")

        df, _source = result
        if df is None or df.empty:
            return self._not_triggered(rule, None, "No daily volume data available")
        if "volume" not in df:
            return self._evaluation_error(rule, "daily data missing volume column")

        try:
            avg_vol = float(df["volume"].mean())
            latest_vol = float(df["volume"].iloc[-1])
        except (TypeError, ValueError, IndexError) as exc:
            return self._evaluation_error(rule, exc)
        if avg_vol <= 0:
            return self._not_triggered(rule, latest_vol, "Average volume is not available")

        ratio = latest_vol / avg_vol
        if latest_vol > avg_vol * rule.multiplier:
            return self._triggered(
                rule,
                latest_vol,
                f"{rule.stock_code} volume spike: {latest_vol:,.0f} ({ratio:.1f}x avg)",
            )
        return self._not_triggered(
            rule,
            latest_vol,
            f"{rule.stock_code} volume ratio {ratio:.1f}x did not exceed {rule.multiplier}x",
        )

    def _triggered(self, rule, observed_value: Any, message: str) -> Dict[str, Any]:
        return {
            "rule_id": self._runtime_rule_id(rule),
            "status": "triggered",
            "triggered": True,
            "observed_value": observed_value,
            "message": self._sanitize_text(message),
        }

    def _not_triggered(self, rule, observed_value: Any, message: str) -> Dict[str, Any]:
        return {
            "rule_id": self._runtime_rule_id(rule),
            "status": "not_triggered",
            "triggered": False,
            "observed_value": observed_value,
            "message": self._sanitize_text(message),
        }

    def _evaluation_error(self, rule, exc: Any) -> Dict[str, Any]:
        return {
            "rule_id": self._runtime_rule_id(rule),
            "status": "evaluation_error",
            "triggered": False,
            "observed_value": None,
            "message": self._sanitize_text(str(exc) or "Alert evaluation failed"),
        }

    @staticmethod
    def _runtime_rule_id(rule) -> int:
        return int(rule.metadata.get("persisted_rule_id", 0) or 0)

    def list_triggers(
        self,
        *,
        rule_id: Optional[int] = None,
        target: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        rows, total = self.repo.list_triggers(
            rule_id=rule_id,
            target=target,
            status=status,
            page=page,
            page_size=page_size,
        )
        return {
            "items": [self._serialize_trigger(row) for row in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def list_notifications(
        self,
        *,
        trigger_id: Optional[int] = None,
        channel: Optional[str] = None,
        success: Optional[bool] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        rows, total = self.repo.list_notifications(
            trigger_id=trigger_id,
            channel=channel,
            success=success,
            page=page,
            page_size=page_size,
        )
        return {
            "items": [self._serialize_notification(row) for row in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def _normalize_rule_payload(self, payload: Dict[str, Any], *, source: str = "api") -> Dict[str, Any]:
        target_scope = str(payload.get("target_scope") or "single_symbol").strip()
        if target_scope not in SUPPORTED_TARGET_SCOPES:
            raise AlertServiceError(f"unsupported target_scope: {target_scope}")

        target = str(payload.get("target") or "").strip()
        if not target:
            raise AlertServiceError("target is required")

        alert_type = str(payload.get("alert_type") or "").strip().lower()
        if alert_type not in SUPPORTED_ALERT_TYPES:
            raise UnsupportedAlertTypeError(
                f"unsupported alert_type for P1 Alert API: {alert_type or '<empty>'}"
            )

        severity = str(payload.get("severity") or "warning").strip().lower()
        if severity not in SUPPORTED_SEVERITIES:
            raise AlertServiceError(f"unsupported severity: {severity}")

        parameters = self._normalize_parameters(alert_type, payload.get("parameters") or {})
        serialized_rule = {"stock_code": target, "alert_type": alert_type, **parameters}
        try:
            validate_event_alert_rule(serialized_rule)
        except ValueError as exc:
            raise AlertServiceError(str(exc)) from exc

        name = str(payload.get("name") or "").strip()
        if not name:
            name = self._default_rule_name(target=target, alert_type=alert_type, parameters=parameters)

        return {
            "name": name[:64],
            "target_scope": target_scope,
            "target": target,
            "alert_type": alert_type,
            "parameters": self._dump_json(parameters),
            "severity": severity,
            "enabled": bool(payload.get("enabled", True)),
            "source": str(source or "api")[:16],
            "cooldown_policy": self._dump_json_or_none(payload.get("cooldown_policy")),
            "notification_policy": self._dump_json_or_none(payload.get("notification_policy")),
        }

    def _validate_rule_update_payload(self, payload: Dict[str, Any]) -> None:
        for field_name, value in payload.items():
            if value is None and field_name not in NULLABLE_RULE_UPDATE_FIELDS:
                raise AlertServiceError(f"{field_name} must not be null")

    def _normalize_parameters(self, alert_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(parameters, dict):
            raise AlertServiceError("parameters must be an object")

        if alert_type == "price_cross":
            direction = str(parameters.get("direction") or "above").strip().lower()
            if direction not in {"above", "below"}:
                raise AlertServiceError(f"invalid direction: {direction}")
            return {"direction": direction, "price": self._positive_float(parameters.get("price"), "price")}

        if alert_type == "price_change_percent":
            direction = str(parameters.get("direction") or "up").strip().lower()
            if direction not in {"up", "down"}:
                raise AlertServiceError(f"invalid direction: {direction}")
            return {
                "direction": direction,
                "change_pct": self._positive_float(parameters.get("change_pct"), "change_pct"),
            }

        if alert_type == "volume_spike":
            return {"multiplier": self._positive_float(parameters.get("multiplier"), "multiplier")}

        raise UnsupportedAlertTypeError(f"unsupported alert_type for P1 Alert API: {alert_type}")

    @staticmethod
    def _positive_float(value: Any, field_name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise AlertServiceError(f"invalid {field_name}: {value}") from exc
        if number <= 0:
            raise AlertServiceError(f"{field_name} must be > 0")
        return number

    def _to_runtime_rule(self, row: AlertRuleRecord):
        data = self._serialize_rule(row)
        parameters = data["parameters"]
        if data["alert_type"] == "price_cross":
            return PriceAlert(
                stock_code=data["target"],
                direction=str(parameters["direction"]),
                price=float(parameters["price"]),
                metadata={"persisted_rule_id": data["id"]},
            )
        if data["alert_type"] == "price_change_percent":
            return PriceChangeAlert(
                stock_code=data["target"],
                direction=str(parameters["direction"]),
                change_pct=float(parameters["change_pct"]),
                metadata={"persisted_rule_id": data["id"]},
            )
        if data["alert_type"] == "volume_spike":
            return VolumeAlert(
                stock_code=data["target"],
                multiplier=float(parameters["multiplier"]),
                metadata={"persisted_rule_id": data["id"]},
            )
        raise UnsupportedAlertTypeError(f"unsupported alert_type for P1 Alert API: {data['alert_type']}")

    def _serialize_rule(self, row: AlertRuleRecord) -> Dict[str, Any]:
        return {
            "id": row.id,
            "name": row.name,
            "target_scope": row.target_scope,
            "target": row.target,
            "alert_type": row.alert_type,
            "parameters": self._load_json(row.parameters, default={}),
            "severity": row.severity,
            "enabled": bool(row.enabled),
            "source": row.source,
            "cooldown_policy": self._load_json(row.cooldown_policy, default=None),
            "notification_policy": self._load_json(row.notification_policy, default=None),
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    def _serialize_trigger(self, row: AlertTriggerRecord) -> Dict[str, Any]:
        return {
            "id": row.id,
            "rule_id": row.rule_id,
            "target": row.target,
            "observed_value": row.observed_value,
            "threshold": row.threshold,
            "reason": row.reason,
            "data_source": row.data_source,
            "data_timestamp": row.data_timestamp.isoformat() if row.data_timestamp else None,
            "triggered_at": row.triggered_at.isoformat() if row.triggered_at else None,
            "status": row.status,
            "diagnostics": self._sanitize_text(row.diagnostics) if row.diagnostics else None,
        }

    def _serialize_notification(self, row: AlertNotificationRecord) -> Dict[str, Any]:
        return {
            "id": row.id,
            "trigger_id": row.trigger_id,
            "channel": row.channel,
            "attempt": row.attempt,
            "success": bool(row.success),
            "error_code": row.error_code,
            "retryable": bool(row.retryable),
            "latency_ms": row.latency_ms,
            "diagnostics": self._sanitize_text(row.diagnostics) if row.diagnostics else None,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

    @staticmethod
    def _default_rule_name(*, target: str, alert_type: str, parameters: Dict[str, Any]) -> str:
        if alert_type == "price_cross":
            return f"{target} price {parameters['direction']} {parameters['price']}"
        if alert_type == "price_change_percent":
            return f"{target} change {parameters['direction']} {parameters['change_pct']}%"
        if alert_type == "volume_spike":
            return f"{target} volume spike {parameters['multiplier']}x"
        return f"{target} {alert_type}"

    @staticmethod
    def _dump_json(value: Dict[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _dump_json_or_none(self, value: Optional[Dict[str, Any]]) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise AlertServiceError("policy fields must be objects")
        return self._dump_json(value)

    @staticmethod
    def _load_json(raw: Optional[str], *, default: Any) -> Any:
        if raw is None or raw == "":
            return default
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return default

    @staticmethod
    def _sanitize_text(text: Any) -> str:
        sanitized = str(text or "").strip()
        if not sanitized:
            return ""
        sanitized = re.sub(r"(?i)(bearer\s+)[a-z0-9._\-:]+", r"\1[REDACTED]", sanitized)
        sanitized = re.sub(r"(?i)(token|secret|password|sendkey)([=:]\s*)[^\s,;&]+", r"\1\2[REDACTED]", sanitized)
        sanitized = re.sub(r"https?://[^\s]+", "[REDACTED_URL]", sanitized)
        return " ".join(sanitized.split())[:300]
