from __future__ import annotations

import math
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from verl.utils.unified_logger import UnifiedLoggerClient, get_unified_logger


def _env_truthy(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _env_int(v: str | None, default: int) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(v: str | None, default: float) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _get_env(prefix: str | None, key: str, default: str | None = None) -> str | None:
    """
    Env lookup with optional per-component prefix.

    Examples:
      - key="API_LATENCY_ENABLE"
      - prefix="GEMINI" -> checks GEMINI_API_LATENCY_ENABLE first, then API_LATENCY_ENABLE.
    """
    if prefix:
        v = os.getenv(f"{prefix}_{key}")
        if v is not None:
            return v
    return os.getenv(key, default)


@dataclass(frozen=True)
class ApiLatencyConfig:
    enabled: bool
    per_call: bool
    console: bool
    summary_every_n: int
    summary_every_s: float
    sample_size: int

    @staticmethod
    def from_env(prefix: str | None = None) -> "ApiLatencyConfig":
        enabled = _env_truthy(_get_env(prefix, "API_LATENCY_ENABLE", "0"))
        per_call = _env_truthy(_get_env(prefix, "API_LATENCY_PER_CALL", "0"))
        console = _env_truthy(_get_env(prefix, "API_LATENCY_CONSOLE", "0"))
        summary_every_n = max(1, _env_int(_get_env(prefix, "API_LATENCY_SUMMARY_EVERY_N", "200"), 200))
        summary_every_s = max(0.0, _env_float(_get_env(prefix, "API_LATENCY_SUMMARY_EVERY_S", "30.0"), 30.0))
        sample_size = max(0, _env_int(_get_env(prefix, "API_LATENCY_SAMPLE_SIZE", "5000"), 5000))
        return ApiLatencyConfig(
            enabled=enabled,
            per_call=per_call,
            console=console,
            summary_every_n=summary_every_n,
            summary_every_s=summary_every_s,
            sample_size=sample_size,
        )


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0.0:
        return sorted_vals[0]
    if q >= 1.0:
        return sorted_vals[-1]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return (1.0 - frac) * sorted_vals[lo] + frac * sorted_vals[hi]


class ApiLatencyStats:
    """
    Thread-safe latency statistics with bounded memory (reservoir sampling).

    Logged fields (ms):
      - count, ok, err
      - total_ms, mean_ms, std_ms
      - min_ms, p50_ms, p90_ms, p95_ms, p99_ms, max_ms
      - last_ms, sample_n
    """

    def __init__(
        self,
        *,
        name: str,
        event_prefix: str,
        cfg: Optional[ApiLatencyConfig] = None,
        logger: Optional[UnifiedLoggerClient] = None,
        static_fields: Optional[Dict[str, Any]] = None,
        meta_keys: Optional[Tuple[str, ...]] = None,
    ):
        self.name = str(name)
        self.event_prefix = str(event_prefix).strip(".")
        self.cfg = cfg or ApiLatencyConfig.from_env()
        self.logger = logger or get_unified_logger()
        self.static_fields = dict(static_fields or {})
        self._meta_keys = tuple(meta_keys or ())

        self._lock = threading.Lock()
        self._count = 0
        self._ok = 0
        self._err = 0
        self._sum = 0.0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = float("inf")
        self._max = 0.0
        self._last = 0.0
        self._sample: List[float] = []
        self._since_last_log = 0
        self._start_mono = time.monotonic()
        self._last_log_mono = self._start_mono
        self._last_meta: Dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def observe(self, seconds: float, *, ok: bool = True, **fields: Any) -> None:
        if not self.cfg.enabled:
            return

        sec = float(seconds) if seconds is not None else 0.0
        now = time.monotonic()

        call_ev: Optional[Dict[str, Any]] = None
        stats_payload: Optional[Dict[str, Any]] = None

        with self._lock:
            self._count += 1
            self._since_last_log += 1
            if ok:
                self._ok += 1
            else:
                self._err += 1

            self._sum += sec
            self._last = sec
            if sec < self._min:
                self._min = sec
            if sec > self._max:
                self._max = sec

            # Welford running variance
            delta = sec - self._mean
            self._mean += delta / self._count
            delta2 = sec - self._mean
            self._m2 += delta * delta2

            # Reservoir sampling for quantiles
            k = int(self.cfg.sample_size)
            if k > 0:
                if len(self._sample) < k:
                    self._sample.append(sec)
                else:
                    j = random.randrange(self._count)
                    if j < k:
                        self._sample[j] = sec

            if self.cfg.per_call:
                call_ev = {
                    "event_type": f"{self.event_prefix}.call",
                    "name": self.name,
                    "ok": bool(ok),
                    "latency_ms": sec * 1000.0,
                    **self.static_fields,
                    **fields,
                }

            if self._meta_keys:
                for k in self._meta_keys:
                    if k in fields:
                        self._last_meta[k] = fields[k]

            should_log_n = self._since_last_log >= int(self.cfg.summary_every_n)
            should_log_s = (
                float(self.cfg.summary_every_s) > 0.0
                and (now - self._last_log_mono) >= float(self.cfg.summary_every_s)
            )
            if should_log_n or should_log_s:
                stats_payload = self._snapshot_locked(now)
                self._since_last_log = 0
                self._last_log_mono = now

        if call_ev is not None:
            self._emit_event(call_ev)

        if stats_payload is not None:
            self._emit_summary(stats_payload)

    def log_summary(self, **fields: Any) -> None:
        if not self.cfg.enabled:
            return
        now = time.monotonic()
        with self._lock:
            payload = self._snapshot_locked(now)
            self._since_last_log = 0
            self._last_log_mono = now
        self._emit_summary(payload)

    def _snapshot_locked(self, now_mono: float) -> Dict[str, Any]:
        count = int(self._count)
        ok_n = int(self._ok)
        err_n = int(self._err)
        total_s = float(self._sum)
        mean_s = float(self._mean) if count > 0 else 0.0
        std_s = math.sqrt(float(self._m2) / (count - 1)) if count > 1 else 0.0
        min_s = float(self._min) if count > 0 and self._min != float("inf") else 0.0
        max_s = float(self._max) if count > 0 else 0.0
        last_s = float(self._last) if count > 0 else 0.0
        sample = list(self._sample)

        q = {}
        if sample:
            sample.sort()
            q = {
                "p50_ms": _quantile(sample, 0.50) * 1000.0,
                "p90_ms": _quantile(sample, 0.90) * 1000.0,
                "p95_ms": _quantile(sample, 0.95) * 1000.0,
                "p99_ms": _quantile(sample, 0.99) * 1000.0,
            }

        elapsed_s = max(0.0, float(now_mono - self._start_mono))
        rps = (count / elapsed_s) if elapsed_s > 0 else 0.0

        payload: Dict[str, Any] = {
            "event_type": f"{self.event_prefix}.stats",
            "name": self.name,
            "count": count,
            "ok": ok_n,
            "err": err_n,
            "sample_n": len(sample),
            "elapsed_s": elapsed_s,
            "rps": rps,
            "total_ms": total_s * 1000.0,
            "mean_ms": mean_s * 1000.0,
            "std_ms": std_s * 1000.0,
            "min_ms": min_s * 1000.0,
            "max_ms": max_s * 1000.0,
            "last_ms": last_s * 1000.0,
            **q,
            **self.static_fields,
            **self._last_meta,
        }
        return payload

    def _emit_event(self, payload: Dict[str, Any]) -> None:
        if self.logger is not None and getattr(self.logger, "enabled", False):
            try:
                self.logger.log_event(payload)
            except Exception:
                pass
        if self.cfg.console:
            try:
                ok = payload.get("ok", None)
                latency_ms = payload.get("latency_ms", None)
                name = payload.get("name", self.name)
                print(f"[API-LATENCY][CALL] {name} ok={ok} latency_ms={latency_ms:.1f}")
            except Exception:
                pass

    def _emit_summary(self, payload: Dict[str, Any]) -> None:
        if self.logger is not None and getattr(self.logger, "enabled", False):
            try:
                self.logger.log_event(payload)
            except Exception:
                pass

        if not self.cfg.console:
            return

        try:
            name = payload.get("name", self.name)
            count = int(payload.get("count", 0) or 0)
            ok_n = int(payload.get("ok", 0) or 0)
            err_n = int(payload.get("err", 0) or 0)
            mean_ms = float(payload.get("mean_ms", 0.0) or 0.0)
            p50_ms = float(payload.get("p50_ms", float("nan")))
            p95_ms = float(payload.get("p95_ms", float("nan")))
            p99_ms = float(payload.get("p99_ms", float("nan")))
            max_ms = float(payload.get("max_ms", 0.0) or 0.0)
            total_ms = float(payload.get("total_ms", 0.0) or 0.0)
            min_ms = float(payload.get("min_ms", 0.0) or 0.0)
            std_ms = float(payload.get("std_ms", 0.0) or 0.0)

            def _fmt(x: float) -> str:
                return "nan" if isinstance(x, float) and math.isnan(x) else f"{x:.1f}"

            print(
                "[API-LATENCY][STATS] "
                f"{name} n={count} ok={ok_n} err={err_n} "
                f"mean={mean_ms:.1f}ms std={std_ms:.1f}ms "
                f"min={min_ms:.1f}ms p50={_fmt(p50_ms)}ms p95={_fmt(p95_ms)}ms p99={_fmt(p99_ms)}ms "
                f"max={max_ms:.1f}ms total={total_ms/1000.0:.2f}s"
            )
        except Exception:
            return
