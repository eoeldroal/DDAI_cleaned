from __future__ import annotations

import json
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _env_truthy(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _now_ts() -> float:
    return time.time()


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps({"event_type": "unified_logger.json_dumps_failed", "ts": _now_ts()}, ensure_ascii=False)


@dataclass(frozen=True)
class UnifiedLoggerConfig:
    enabled: bool
    log_path: str
    actor_name: str
    client_batch_size: int
    client_flush_interval_s: float
    writer_flush_every_n: int
    writer_flush_interval_s: float


def get_unified_logger_config() -> UnifiedLoggerConfig:
    enabled = _env_truthy("UNIFIED_LOG_ENABLE", "0")
    log_path = os.getenv("UNIFIED_LOG_PATH", os.path.join("./logs", "unified_trajectory.jsonl"))
    actor_name = os.getenv("UNIFIED_LOG_ACTOR_NAME", f"unified_log_writer_{os.getpid()}")
    client_batch_size = int(os.getenv("UNIFIED_LOG_CLIENT_BATCH_SIZE", "200"))
    client_flush_interval_s = float(os.getenv("UNIFIED_LOG_CLIENT_FLUSH_INTERVAL_S", "1.0"))
    writer_flush_every_n = int(os.getenv("UNIFIED_LOG_WRITER_FLUSH_EVERY_N", "2000"))
    writer_flush_interval_s = float(os.getenv("UNIFIED_LOG_WRITER_FLUSH_INTERVAL_S", "2.0"))
    return UnifiedLoggerConfig(
        enabled=enabled,
        log_path=log_path,
        actor_name=actor_name,
        client_batch_size=client_batch_size,
        client_flush_interval_s=client_flush_interval_s,
        writer_flush_every_n=writer_flush_every_n,
        writer_flush_interval_s=writer_flush_interval_s,
    )


def _base_context() -> Dict[str, Any]:
    return {
        "ts": _now_ts(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }


def _ensure_log_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _get_ray():
    try:
        import ray  # type: ignore
    except Exception:
        return None
    return ray


def _get_or_create_writer_actor(cfg: UnifiedLoggerConfig):
    ray = _get_ray()
    if ray is None:
        return None
    if not ray.is_initialized():
        return None

    @ray.remote(num_cpus=0)
    class UnifiedLogWriter:
        def __init__(
            self,
            log_path: str,
            flush_every_n: int = 2000,
            flush_interval_s: float = 2.0,
        ):
            self.log_path = log_path
            self.flush_every_n = max(int(flush_every_n), 1)
            self.flush_interval_s = max(float(flush_interval_s), 0.05)
            _ensure_log_dir(self.log_path)
            self._f = open(self.log_path, "a", encoding="utf-8")
            self._buf: List[str] = []
            self._last_flush = time.monotonic()

        def log_many(self, events: List[Dict[str, Any]]) -> None:
            now = time.monotonic()
            for e in events:
                self._buf.append(_safe_json_dumps(e))
            if len(self._buf) >= self.flush_every_n or (now - self._last_flush) >= self.flush_interval_s:
                self._flush_locked(now)

        def flush(self) -> None:
            self._flush_locked(time.monotonic())

        def close(self) -> None:
            self._flush_locked(time.monotonic())
            try:
                self._f.close()
            except Exception:
                pass

        def _flush_locked(self, now: float) -> None:
            if not self._buf:
                self._last_flush = now
                return
            try:
                self._f.write("\n".join(self._buf) + "\n")
                self._f.flush()
            except Exception:
                # Best-effort: never crash training for logging.
                pass
            finally:
                self._buf.clear()
                self._last_flush = now

    try:
        return ray.get_actor(cfg.actor_name)
    except Exception:
        try:
            return UnifiedLogWriter.options(name=cfg.actor_name).remote(
                cfg.log_path,
                flush_every_n=cfg.writer_flush_every_n,
                flush_interval_s=cfg.writer_flush_interval_s,
            )
        except Exception:
            return None


class _LocalUnifiedLogWriter:
    def __init__(self, log_path: str, flush_every_n: int, flush_interval_s: float):
        self.log_path = log_path
        self.flush_every_n = max(int(flush_every_n), 1)
        self.flush_interval_s = max(float(flush_interval_s), 0.05)
        _ensure_log_dir(self.log_path)
        self._f = open(self.log_path, "a", encoding="utf-8")
        self._buf: List[str] = []
        self._last_flush = time.monotonic()
        self._lock = threading.Lock()

    def log_many(self, events: List[Dict[str, Any]]) -> None:
        now = time.monotonic()
        with self._lock:
            for e in events:
                self._buf.append(_safe_json_dumps(e))
            if len(self._buf) >= self.flush_every_n or (now - self._last_flush) >= self.flush_interval_s:
                self._flush_locked(now)

    def flush(self) -> None:
        with self._lock:
            self._flush_locked(time.monotonic())

    def close(self) -> None:
        with self._lock:
            self._flush_locked(time.monotonic())
            try:
                self._f.close()
            except Exception:
                pass

    def _flush_locked(self, now: float) -> None:
        if not self._buf:
            self._last_flush = now
            return
        try:
            self._f.write("\n".join(self._buf) + "\n")
            self._f.flush()
        except Exception:
            pass
        finally:
            self._buf.clear()
            self._last_flush = now


class UnifiedLoggerClient:
    def __init__(self, cfg: UnifiedLoggerConfig):
        self.cfg = cfg
        self._writer = None
        self._lock = threading.Lock()
        self._buf: List[Dict[str, Any]] = []
        self._last_send = time.monotonic()
        self._static_context = _base_context()

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def set_static_fields(self, **fields: Any) -> None:
        with self._lock:
            self._static_context.update(fields)

    def log(self, event_type: str, **fields: Any) -> None:
        if not self.cfg.enabled:
            return
        ev = dict(self._static_context)
        ev["ts"] = _now_ts()
        ev["event_type"] = event_type
        ev.update(fields)
        with self._lock:
            self._buf.append(ev)
            self._maybe_flush_locked()

    def log_event(self, event: Dict[str, Any]) -> None:
        if not self.cfg.enabled:
            return
        ev = dict(self._static_context)
        # Ensure per-event timestamps even though static context includes an init-time `ts`.
        # If the caller provides an explicit `ts`, it should override this value.
        ev["ts"] = _now_ts()
        ev.update(event)
        ev.setdefault("event_type", "event")
        with self._lock:
            self._buf.append(ev)
            self._maybe_flush_locked()

    def flush(self) -> None:
        if not self.cfg.enabled:
            return
        with self._lock:
            self._flush_locked(force=True)

    def _maybe_flush_locked(self) -> None:
        now = time.monotonic()
        if len(self._buf) >= self.cfg.client_batch_size or (now - self._last_send) >= self.cfg.client_flush_interval_s:
            self._flush_locked(force=False)

    def _flush_locked(self, force: bool) -> None:
        if not self._buf:
            return
        if self._writer is None:
            self._writer = _get_or_create_writer_actor(self.cfg)
            if self._writer is None:
                self._writer = _LocalUnifiedLogWriter(
                    self.cfg.log_path,
                    flush_every_n=self.cfg.writer_flush_every_n,
                    flush_interval_s=self.cfg.writer_flush_interval_s,
                )
        if self._writer is None:
            if force:
                self._buf.clear()
            return
        batch = self._buf
        self._buf = []
        self._last_send = time.monotonic()
        try:
            remote_fn = getattr(getattr(self._writer, "log_many", None), "remote", None)
            if callable(remote_fn):
                remote_fn(batch)
            else:
                self._writer.log_many(batch)
            if force:
                flush_remote = getattr(getattr(self._writer, "flush", None), "remote", None)
                if callable(flush_remote):
                    flush_remote()
                else:
                    flush_local = getattr(self._writer, "flush", None)
                    if callable(flush_local):
                        flush_local()
        except Exception:
            if force:
                return
            # On transient failure, drop rather than blocking training.
            return


_UNIFIED_LOGGER_SINGLETON: Optional[UnifiedLoggerClient] = None
_UNIFIED_LOGGER_SINGLETON_LOCK = threading.Lock()


def get_unified_logger() -> UnifiedLoggerClient:
    global _UNIFIED_LOGGER_SINGLETON
    with _UNIFIED_LOGGER_SINGLETON_LOCK:
        if _UNIFIED_LOGGER_SINGLETON is None:
            cfg = get_unified_logger_config()
            _UNIFIED_LOGGER_SINGLETON = UnifiedLoggerClient(cfg)
        return _UNIFIED_LOGGER_SINGLETON
