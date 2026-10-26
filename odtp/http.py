"""Disk-cached HTTP client with a pause between calls and retries.

Cache files live under ``data/cache/`` and are keyed by the method, URL, query
string, and body. Failed responses are not cached. A cache hit does not count
toward the rate limit.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from odtp.config import USER_AGENT

logger = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}
SleepFn = Callable[[float], None]
TimeFn = Callable[[], float]


class RequestError(RuntimeError):
    """A request failed after retries, or the server returned a hard error."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class HttpClient:
    """Small wrapper around ``requests`` used by every scientific API client."""

    def __init__(
        self,
        cache_dir: Path | None,
        *,
        session: Any | None = None,
        min_interval: float = 0.2,
        max_retries: int = 4,
        timeout: float = 60.0,
        sleep: SleepFn | None = None,
        time_fn: TimeFn | None = None,
    ) -> None:
        import time

        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.sleep = sleep or time.sleep
        self.time = time_fn or time.monotonic
        self._last: float | None = None
        self.network_calls = 0
        self.cache_hits = 0

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", url, params=params, body=None, extra_headers=None)

    def post_json(self, url: str, payload: dict[str, Any]) -> Any:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return self._request(
            "POST",
            url,
            params=None,
            body=body,
            extra_headers={"Content-Type": "application/json"},
        )

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None,
        body: bytes | None,
        extra_headers: dict[str, str] | None,
    ) -> Any:
        key = self._cache_key(method, url, params, body)
        cached = self._read_cache(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._rate_limit()
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    data=body,
                    headers=headers,
                    timeout=self.timeout,
                )
                self.network_calls += 1
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("%s %s failed (%s); retrying", method, url, exc)
                self._backoff(attempt)
                continue
            if response.status_code in RETRY_STATUS:
                last_error = RequestError(
                    f"{response.status_code} from {url}", status=response.status_code
                )
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    self.sleep(float(retry_after))
                else:
                    self._backoff(attempt)
                continue
            if response.status_code != 200:
                raise RequestError(
                    f"{response.status_code} from {url}", status=response.status_code
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise RequestError(f"Invalid JSON from {url}") from exc
            if isinstance(payload, dict) and payload.get("errors") and not payload.get("data"):
                raise RequestError(f"API error from {url}: {payload['errors'][:1]}")
            self._write_cache(key, payload)
            return payload
        raise RequestError(f"Failed after {self.max_retries} attempts for {url}: {last_error}")

    def _rate_limit(self) -> None:
        now = self.time()
        if self._last is None:
            self._last = now
            return
        wait = self.min_interval - (now - self._last)
        if wait > 0:
            self.sleep(wait)
        self._last = self.time()

    def _backoff(self, attempt: int) -> None:
        self.sleep(min(8.0, 0.8 * (2**attempt)))

    def _cache_key(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        body: bytes | None,
    ) -> str:
        raw = json.dumps(
            {
                "method": method,
                "url": url,
                "params": params or {},
                "body": body.decode() if body else "",
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> Any | None:
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        return json.loads(path.read_text())

    def _write_cache(self, key: str, payload: Any) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        path.write_text(json.dumps(payload))
